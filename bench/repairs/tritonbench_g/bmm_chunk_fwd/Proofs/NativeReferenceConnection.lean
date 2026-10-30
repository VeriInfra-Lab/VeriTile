import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.OrderedNative
import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.NativeGeometry

namespace VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO.Native
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open KernelInterface NativeGraph
noncomputable section
open Classical
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

abbrev Memory := RegionName → Nat → Cell Expr

/-- Supporting representation of independent ordered output writes. It never
appears in the independent public numerical formula. -/
def patch (p : Parameters) (initial : Memory) (writes : Writes) : Memory := fun r off=>
  if r=p.Out then match writes off with
    | some x=>.mk .fp32 x
    | none=>initial r off
  else initial r off

structure Reads (p : Parameters) (c : Host.Config) (v : Values p) (initial : Memory) : Prop where
  a : ∀ off,readMask p c 0 off → (initial p.A off).read .real=a p v off
  b : ∀ off,readMask p c 1 off → (initial p.B off).read .real=b p v off
  seq : ∀ off,readMask p c 2 off → (initial (Region.cast p.Seq) off).read .int=seq p v off
  out : ∀ off,off<p.shape.outputSize → (initial p.Out off).read .fp32=oldOutput p v off

theorem read_extent (p : Parameters) (c : Host.Config) (which : Fin 3) (off : Nat)
    (active : readMask p c which off) : off < inputArity p which := by
  rcases which with ⟨which,lt⟩
  interval_cases which
  · obtain ⟨_,pid,_,q,i,e,active,rfl⟩ := active
    exact a_address_bound p c pid q i e active
  · obtain ⟨_,pid,_,q,e,j,active,rfl⟩ := active
    exact b_address_bound p c pid q e j active
  · obtain ⟨_,_,pid,_,active⟩ := active
    rcases active with ⟨i,active,rfl⟩|⟨j,active,rfl⟩
    · exact sequence_address_bound p c pid _ active
    · exact sequence_address_bound p c pid _ active

theorem reads_from_binding (rest : Algebra Expr) (p : Parameters) (c : Host.Config)
    (v : Values p) (s : State Expr) (binding : (io rest p c).InputBinding v s) : Reads p c v s.mem := by
  constructor
  · intro off active
    have bound := read_extent p c 0 off active
    have pin := binding (0 : Fin 4) (⟨off,bound⟩,PUnit.unit) active
    simpa [a,show off<dataExtent p from bound,ProgramIO.InputBinding,mapValue] using pin
  · intro off active
    have bound := read_extent p c 1 off active
    have pin := binding (1 : Fin 4) (⟨off,bound⟩,PUnit.unit) active
    simpa [b,show off<dataExtent p from bound,ProgramIO.InputBinding,mapValue] using pin
  · intro off active
    have bound := read_extent p c 2 off active
    have pin := binding (2 : Fin 4) (⟨off,bound⟩,PUnit.unit) active
    simpa [seq,show off<seqExtent p from bound,ProgramIO.InputBinding,mapValue] using pin
  · intro off bound
    have pin := binding (3 : Fin 4) (⟨off,bound⟩,PUnit.unit) trivial
    simpa [oldOutput,bound,ProgramIO.InputBinding,mapValue] using pin

theorem patch_a (p : Parameters) (c : Host.Config) (v : Values p) (initial : Memory)
    (reads : Reads p c v initial) (writes : Writes) (off : Nat) (active : readMask p c 0 off) :
    (patch p initial writes p.A off).read .real=(operands p v writes).a off := by
  have read := reads.a off active
  by_cases same : p.A=p.Out <;> cases hw : writes off <;>
    simp only [patch,same,hw,↓reduceIte]
  all_goals first | simpa [operands,same,hw] using read | simp [operands,same,hw,Cell.read,defaultValue]

theorem patch_b (p : Parameters) (c : Host.Config) (v : Values p) (initial : Memory)
    (reads : Reads p c v initial) (writes : Writes) (off : Nat) (active : readMask p c 1 off) :
    (patch p initial writes p.B off).read .real=(operands p v writes).b off := by
  have read := reads.b off active
  by_cases same : p.B=p.Out <;> cases hw : writes off <;>
    simp only [patch,same,hw,↓reduceIte]
  all_goals first | simpa [operands,same,hw] using read | simp [operands,same,hw,Cell.read,defaultValue]

theorem patch_seq (p : Parameters) (c : Host.Config) (v : Values p) (initial : Memory)
    (reads : Reads p c v initial) (writes : Writes) (off : Nat) (active : readMask p c 2 off) :
    (patch p initial writes (Region.cast p.Seq) off).read .int=(operands p v writes).seq off := by
  have read := reads.seq off active
  by_cases same : Region.cast p.Seq=p.Out <;> cases hw : writes off <;>
    simp only [patch,same,hw,↓reduceIte]
  all_goals first | simpa [operands,same,hw] using read | simp [operands,same,hw,Cell.read,defaultValue]

theorem aTile_eq (p : Parameters) (c : Host.Config) (v : Values p) (initial : Memory)
    (reads : Reads p c v initial) (writes : Writes) (pid : Host.PID (parameters p) c)
    (nonempty : p.shape.outputSize≠0)
    (active : NativeReference.activeBlock (parameters p) c (coordinates p c pid)=Bool.true)
    (q : Nat) (bound : q<Host.cdiv p.shape.K c.K) :
    NativeReference.aTile (parameters p) c (coordinates p c pid)
      (NativeReference.capture (parameters p) (patch p initial writes)) q=
    NativeReference.aTile (parameters p) c (coordinates p c pid) (operands p v writes) q := by
  funext i
  unfold NativeReference.aTile
  split_ifs with mask
  · rw [show (NativeReference.capture (parameters p) (patch p initial writes)).a _=
        (operands p v writes).a _ from
      patch_a p c v initial reads writes _ ⟨nonempty,pid,active,⟨q,bound⟩,i.1,i.2.1,mask,rfl⟩]
  · rfl

theorem bTile_eq (p : Parameters) (c : Host.Config) (v : Values p) (initial : Memory)
    (reads : Reads p c v initial) (writes : Writes) (pid : Host.PID (parameters p) c)
    (nonempty : p.shape.outputSize≠0)
    (active : NativeReference.activeBlock (parameters p) c (coordinates p c pid)=Bool.true)
    (q : Nat) (bound : q<Host.cdiv p.shape.K c.K) :
    NativeReference.bTile (parameters p) c (coordinates p c pid)
      (NativeReference.capture (parameters p) (patch p initial writes)) q=
    NativeReference.bTile (parameters p) c (coordinates p c pid) (operands p v writes) q := by
  funext i
  unfold NativeReference.bTile
  split_ifs with mask
  · rw [show (NativeReference.capture (parameters p) (patch p initial writes)).b _=
        (operands p v writes).b _ from
      patch_b p c v initial reads writes _ ⟨nonempty,pid,active,⟨q,bound⟩,i.1,i.2.1,mask,rfl⟩]
  · rfl

theorem accumulator_eq (p : Parameters) (c : Host.Config) (v : Values p) (initial : Memory)
    (reads : Reads p c v initial) (writes : Writes) (pid : Host.PID (parameters p) c)
    (nonempty : p.shape.outputSize≠0)
    (active : NativeReference.activeBlock (parameters p) c (coordinates p c pid)=Bool.true)
    (q : Nat) (bound : q≤Host.cdiv p.shape.K c.K) :
    NativeReference.accumulator (parameters p) c (coordinates p c pid)
      (NativeReference.capture (parameters p) (patch p initial writes)) q=
    NativeReference.accumulator (parameters p) c (coordinates p c pid) (operands p v writes) q := by
  induction q with
  | zero => rfl
  | succ q ih =>
    funext i
    simp only [NativeReference.accumulator,
      aTile_eq p c v initial reads writes pid nonempty active q (by omega),
      bTile_eq p c v initial reads writes pid nonempty active q (by omega),ih (by omega)]

theorem output_eq (p : Parameters) (c : Host.Config) (v : Values p) (initial : Memory)
    (reads : Reads p c v initial) (writes : Writes) (pid : Host.PID (parameters p) c)
    (nonempty : p.shape.outputSize≠0)
    (active : NativeReference.activeBlock (parameters p) c (coordinates p c pid)=Bool.true)
    (i : TileIndex [c.M,c.N]) :
    NativeReference.output (parameters p) c (coordinates p c pid)
      (NativeReference.capture (parameters p) (patch p initial writes)) i=
    NativeReference.output (parameters p) c (coordinates p c pid) (operands p v writes) i := by
  have acc := congrFun (accumulator_eq p c v initial reads writes pid nonempty active
    (Host.cdiv p.shape.K c.K) (Nat.le_refl _)) i
  unfold NativeReference.output
  rw [show (parameters p).shape.K=p.shape.K from rfl,acc]
  cases hs : p.hasSeq
  · simp [parameters,hs]
  · have row : NativeReference.sequence (parameters p) (coordinates p c pid)
        (NativeReference.capture (parameters p) (patch p initial writes))
        (SourcePointers.row c (coordinates p c pid) i.1) (-1)=
      NativeReference.sequence (parameters p) (coordinates p c pid) (operands p v writes)
        (SourcePointers.row c (coordinates p c pid) i.1) (-1) := by
      unfold NativeReference.sequence
      split_ifs with loaded
      · exact patch_seq p c v initial reads writes _
          ⟨nonempty,hs,pid,active,Or.inl ⟨i.1,loaded,rfl⟩⟩
      · rfl
    have col : NativeReference.sequence (parameters p) (coordinates p c pid)
        (NativeReference.capture (parameters p) (patch p initial writes))
        (SourcePointers.col c (coordinates p c pid) i.2.1) (-2)=
      NativeReference.sequence (parameters p) (coordinates p c pid) (operands p v writes)
        (SourcePointers.col c (coordinates p c pid) i.2.1) (-2) := by
      unfold NativeReference.sequence
      split_ifs with loaded
      · exact patch_seq p c v initial reads writes _
          ⟨nonempty,hs,pid,active,Or.inr ⟨i.2.1,loaded,rfl⟩⟩
      · rfl
    rw [row,col]

end
end VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO.Native
