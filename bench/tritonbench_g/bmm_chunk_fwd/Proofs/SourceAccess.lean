import bench.tritonbench_g.bmm_chunk_fwd.Proofs.SourcePointers

namespace VeriTile.Bench.Repairs.BmmChunkFwd.SourceAccess
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open FPSuccess SourcePointers
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def aMask (h : Host.Call) (c : Host.Config) (d : Coordinates) (q i e : Nat) : Bool :=
  decide (row c d i<limit h d) && decide (e<h.shape.K-q*c.K)
def bMask (h : Host.Call) (c : Host.Config) (d : Coordinates) (q e j : Nat) : Bool :=
  decide (e<h.shape.K-q*c.K) && decide (col c d j<limit h d)

def aValue {α : Type} [Inhabited α] (o : Ops α) (h : Host.Call) (c : Host.Config) (d : Coordinates)
    (q : Nat) (s : State α) (i e : Nat) : α :=
  o.scalar.cast (some .fp32) .real .real
    (if aMask h c d q i e then (s.mem h.A (aOffset h c d q i e)).read .real
      else o.scalar.literal (some .fp32) .real 0)
def bValue {α : Type} [Inhabited α] (o : Ops α) (h : Host.Call) (c : Host.Config) (d : Coordinates)
    (q : Nat) (s : State α) (e j : Nat) : α :=
  o.scalar.cast (some .fp32) .real .real
    (if bMask h c d q e j then (s.mem h.B (bOffset h c d q e j)).read .real
      else o.scalar.literal (some .fp32) .real 0)

/-- Exact loaded tile registers of the actual full K body, retaining its casts.
The conditional is outside the memory read, so masked lanes do not dereference
whatever modular pointer the source formed for them. -/
theorem loads {α : Type} [Inhabited α] (o : Ops α) (h : Host.Call) (c : Host.Config)
    (d : Coordinates) (q : Nat) (s : State α) (hs : StateAt h c d q s)
    (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32) :
    ∃ t, run (engine o.algebra) (parts h c).body
      (s.setReg "k" .nat [] (fun _ => q))=some t ∧
      t.regs .real [c.M,c.K] "a"=some (fun i => aValue o h c d q s i.1 i.2.1) ∧
      t.regs .real [c.K,c.N] "b"=some (fun i => bValue o h c d q s i.1 i.2.1) := by
  rcases hs with ⟨⟨h0,h1,h2,h3,h4⟩,h5,h6,h7,h8,h9,h10,h11,hseq⟩
  obtain ⟨acc,h11⟩ := Option.isSome_iff_exists.mp h11
  simp (config := {maxSteps := 400000}) [parts,branch,Kernels.source,Host.Call.kernel,TypedSource.specialize,VeriTile.Bench.Repairs.BmmChunkFwd.Kernels.mainSource,
    ha,hb,Source.conditional?,Source.dynamicRange?,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
    Algebra.withDefaultPrecision,resolvePrecision,Ops.algebra,ComputeDType.eraseDType,
    numeric,natLt,natLe,numericLt,numericLe,bop,ofFloat,toFloat,TileShape.axisDim,
    Region.cast,State.setReg,integralFloorDiv,integralMod,
    h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,aValue,bValue,aMask,bMask]
  constructor <;> funext i <;> norm_num

def seqValue {α : Type} [Inhabited α] (h : Host.Call) (d : Coordinates) (s : State α)
    (lane : Nat) (fallback : Int) : Int :=
  if lane<limit h d then (s.mem (Region.cast h.Seq) (seqOffset h d lane)).read .int else fallback

def outMask (h : Host.Call) (c : Host.Config) (d : Coordinates) (i j : Nat) : Bool :=
  decide (row c d i<h.shape.chunk) && decide (col c d j<h.shape.chunk)

@[simp] theorem store_regs {α ι : Type} (r : ι → RegionName) (off : ι → Nat)
    (val : ι → Cell α) (active : ι → Prop) [DecidablePred active] (l : List ι) (s : State α) :
    (l.foldl (fun t i => if active i then t.write (r i) (off i) (val i) else t) s).regs=s.regs := by
  induction l generalizing s with
  | nil => rfl
  | cons i rest ih =>
    rw [List.foldl_cons,ih]
    split <;> rfl

/-- Actual epilogue execution keeps the exact source output pointers and the
signed Seq loaded registers, including the distinct -1 and -2 masked fallbacks. -/
theorem epilogue {α : Type} [Inhabited α] (o : Ops α) (h : Host.Call) (c : Host.Config)
    (d : Coordinates) (q : Nat) (s : State α) (hs : StateAt h c d q s)
    (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32) :
    ∃ t, run (engine o.algebra) (parts h c).after s=some t ∧
      t.regs .ptr [c.M,c.N] "out_ptrs"=some (fun i => (h.Out,outOffset h c d i.1 i.2.1)) ∧
      (h.hasSeq=Bool.true →
        t.regs .int [c.M] "seq_idx_m"=some (fun i => seqValue h d s (row c d i.1) (-1)) ∧
        t.regs .int [c.N] "seq_idx_n"=some (fun i => seqValue h d s (col c d i.1) (-2))) := by
  rcases hs with ⟨⟨h0,h1,h2,h3,h4⟩,h5,h6,h7,h8,h9,h10,h11,hseq⟩
  obtain ⟨acc,h11⟩ := Option.isSome_iff_exists.mp h11
  cases seq : h.hasSeq
  all_goals
    simp (config := {maxSteps := 400000}) [parts,branch,Kernels.source,Host.Call.kernel,TypedSource.specialize,VeriTile.Bench.Repairs.BmmChunkFwd.Kernels.mainSource,
      ha,hb,seq,Source.conditional?,Source.dynamicRange?,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
      Algebra.withDefaultPrecision,resolvePrecision,Ops.algebra,ComputeDType.eraseDType,
      numeric,natLt,natLe,numericLt,numericLe,numericEq,bop,ofFloat,toFloat,TileShape.axisDim,
      Region.cast,State.setReg,integralFloorDiv,integralMod,
      h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,hseq,FP.Structural.store,store_regs,
      seqValue,seqOffset,seqBase,outOffset,SourcePointers.row,col,limit,minimum]
  all_goals
    repeat' constructor
    all_goals
      funext i
      simp [bop,minimum,Nat.add_mul,Nat.mul_add,Nat.mul_assoc,Nat.add_assoc,
        Nat.add_comm,Nat.add_left_comm]
      all_goals omega

/-- Exact active store-address image in the source epilogue. -/
def Writes (h : Host.Call) (c : Host.Config) (d : Coordinates) (rg : RegionName) (off : Nat) : Prop :=
  ∃ i : TileIndex [c.M,c.N], outMask h c d i.1 i.2.1=Bool.true ∧ rg=h.Out ∧ off=outOffset h c d i.1 i.2.1

theorem epilogue_frame {α : Type} [Inhabited α] (o : Ops α) (h : Host.Call) (c : Host.Config)
    (d : Coordinates) (q : Nat) (s : State α) (hs : StateAt h c d q s)
    (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32) :
    ∃ t, run (engine o.algebra) (parts h c).after s=some t ∧
      (∀ rg off, ¬Writes h c d rg off → t.mem rg off=s.mem rg off) := by
  rcases hs with ⟨⟨h0,h1,h2,h3,h4⟩,h5,h6,h7,h8,h9,h10,h11,hseq⟩
  obtain ⟨acc,h11⟩ := Option.isSome_iff_exists.mp h11
  cases seq : h.hasSeq
  all_goals
    simp (config := {maxSteps := 400000}) [parts,branch,Kernels.source,Host.Call.kernel,TypedSource.specialize,VeriTile.Bench.Repairs.BmmChunkFwd.Kernels.mainSource,
      ha,hb,seq,Source.conditional?,Source.dynamicRange?,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
      Algebra.withDefaultPrecision,resolvePrecision,Ops.algebra,ComputeDType.eraseDType,
      numeric,natLt,natLe,numericLt,numericLe,numericEq,bop,ofFloat,toFloat,TileShape.axisDim,
      Region.cast,State.setReg,integralFloorDiv,integralMod,
      h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,hseq,FP.Structural.store]
  all_goals
    intro rg off miss
    apply State.masked_scatter_frame
    by_cases hr : rg=h.Out
    · right
      intro i _ hi ho
      apply miss
      refine ⟨i,?_,hr,?_⟩
      · simpa [outMask,SourcePointers.row,col] using hi
      · simpa [outOffset,SourcePointers.row,col,mul_ite] using ho
    · exact Or.inl hr

#axiomsClean epilogue_frame
#axiomsClean loads
#axiomsClean epilogue
end
end VeriTile.Bench.Repairs.BmmChunkFwd.SourceAccess

#guard_msgs (drop info) in
#auditModuleAxioms
