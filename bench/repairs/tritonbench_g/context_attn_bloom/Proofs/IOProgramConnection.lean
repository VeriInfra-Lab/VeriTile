import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.IOAccessBounds
import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.IOOwnership

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.KernelIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program
noncomputable section
open Classical
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000
variable {α : Type} [Inhabited α]

def tileActive (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) (xs : Values α r extent)
    (idx : GridIndex (Host.grid g.arguments)) (i : RealIO.Lane (Host.layout g.arguments)) : Prop :=
  RealIO.active (Host.layout g.arguments) (Host.launchOf g.arguments idx) (metadata r extent xs) i

def tileOffset (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) (xs : Values α r extent)
    (idx : GridIndex (Host.grid g.arguments)) (i : RealIO.Lane (Host.layout g.arguments)) : Nat :=
  RealIO.write (Host.layout g.arguments) (Host.launchOf g.arguments idx) (metadata r extent xs) i

theorem active_matches (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (M : Algebra α) (dots : FPSuccess.Dot α) (xs : Values α r extent)
    (valid : domain g r extent xs) (s : State α)
    (hin : (io g r extent M dots).InputBinding xs s) (idx : GridIndex (Host.grid g.arguments))
    (i : RealIO.Lane (Host.layout g.arguments)) :
    FPSuccess.active (Host.layout g.arguments) r ((invocation g r M dots idx).initial s) i ↔
      tileActive g r extent xs idx i := by
  have agree := input_agreement g r extent M dots xs valid s hin idx
  change ((Host.layout g.arguments).BM*(FPReference.inputs (Host.layout g.arguments) r
      ((invocation g r M dots idx).initial s)).block+i.1.val : Int)<
      (FPReference.inputs (Host.layout g.arguments) r ((invocation g r M dots idx).initial s)).length ∧
      i.2.1.val<(Host.layout g.arguments).headDim ↔ _
  rw [agree.block,agree.length]
  simp only [tileActive,RealIO.active,Math.sequence,Math.queryRow,logicalInputs,Nat.cast_add,Nat.cast_mul,Int.mul_comm]

theorem offset_matches (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (M : Algebra α) (dots : FPSuccess.Dot α) (xs : Values α r extent)
    (valid : domain g r extent xs) (s : State α)
    (hin : (io g r extent M dots).InputBinding xs s) (idx : GridIndex (Host.grid g.arguments))
    (i : RealIO.Lane (Host.layout g.arguments)) :
    FPSuccess.offset (Host.layout g.arguments) r ((invocation g r M dots idx).initial s) i=
      tileOffset g r extent xs idx i := by
  have agree := input_agreement g r extent M dots xs valid s hin idx
  change (((FPReference.inputs (Host.layout g.arguments) r ((invocation g r M dots idx).initial s)).start+
    ((Host.layout g.arguments).BM*(FPReference.inputs (Host.layout g.arguments) r
      ((invocation g r M dots idx).initial s)).block+i.1.val : Nat))*(Host.layout g.arguments).outRow+
    ((FPReference.inputs (Host.layout g.arguments) r ((invocation g r M dots idx).initial s)).head*
      (Host.layout g.arguments).outHead : Nat)+(i.2.1.val*(Host.layout g.arguments).outDim : Nat)).toNat=_
  rw [agree.start,agree.block,agree.head]
  simp only [tileOffset,RealIO.write,AddressBounds.outputAddress,Math.queryRow,logicalInputs,Nat.cast_add,Nat.cast_mul,Int.mul_comm]

def tileResult (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (M : Algebra α) (dots : FPSuccess.Dot α) (xs : Values α r extent)
    (idx : GridIndex (Host.grid g.arguments)) : RealIO.Lane (Host.layout g.arguments)→α :=
  FPReference.output (Host.layout g.arguments) M dots (Host.scale g.arguments)
    (logicalInputs g r extent xs idx)

def writeTile (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (M : Algebra α) (dots : FPSuccess.Dot α) (xs : Values α r extent)
    (idx : GridIndex (Host.grid g.arguments)) (s : State α) : State α :=
  (TileShape.allIndices [(Host.layout g.arguments).BM,(Host.layout g.arguments).D]).foldl
    (fun t i=>if tileActive g r extent xs idx i then
      t.write r.out (tileOffset g r extent xs idx i) (.mk .real (tileResult g r extent M dots xs idx i)) else t) s

theorem invocation_correct (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (M : Algebra α) (dots : FPSuccess.Dot α) (xs : Values α r extent)
    (valid : domain g r extent xs) (s : State α)
    (hin : (io g r extent M dots).InputBinding xs s) (idx : GridIndex (Host.grid g.arguments)) :
    ∃ t,(invocation g r M dots idx).run s=some t ∧
      t.mem=(writeTile g r extent M dots xs idx s).mem := by
  have agree := input_agreement g r extent M dots xs valid s hin idx
  obtain ⟨t,ran,memory⟩ := FPReference.program (Host.layout g.arguments) r (Host.scale g.arguments)
    M dots (Host.block_positive g.arguments) ((invocation g r M dots idx).initial s)
  refine ⟨t,ran,?_⟩
  rw [memory,IOReferenceAgreement.output_eq _ r extent M dots _ _ _ agree
    (source_bounds g r extent xs valid idx)]
  simp only [FPSuccess.scatter,writeTile,active_matches g r extent M dots xs valid s hin idx,
    offset_matches g r extent M dots xs valid s hin idx]
  exact FPSuccess.masked_fold_mem_congr r.out (tileOffset g r extent xs idx)
    (fun i=>Cell.mk .real (tileResult g r extent M dots xs idx i))
    (tileActive g r extent xs idx) _ _ _ rfl

theorem writeTile_frame (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (M : Algebra α) (dots : FPSuccess.Dot α) (xs : Values α r extent)
    (idx : GridIndex (Host.grid g.arguments)) (s : State α) (region : RegionName) (off : Nat)
    (outside : ∀ i,tileActive g r extent xs idx i → region≠r.out ∨ off≠tileOffset g r extent xs idx i) :
    (writeTile g r extent M dots xs idx s).mem region off=s.mem region off := by
  apply State.masked_scatter_frame
  by_cases h : region=r.out
  · right
    intro i _ hi
    exact (outside i hi).resolve_left (by simp [h])
  · exact Or.inl h

theorem writeTile_output (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (M : Algebra α) (dots : FPSuccess.Dot α) (xs : Values α r extent)
    (valid : domain g r extent xs) (idx : GridIndex (Host.grid g.arguments))
    (s : State α) (i : RealIO.Lane (Host.layout g.arguments)) (hi : tileActive g r extent xs idx i) :
    (writeTile g r extent M dots xs idx s).mem r.out (tileOffset g r extent xs idx i)=
      .mk .real (tileResult g r extent M dots xs idx i) := by
  obtain ⟨before,after,equal⟩ := List.append_of_mem (TileShape.mem_allIndices
    [(Host.layout g.arguments).BM,(Host.layout g.arguments).D] i)
  have distinct := TileShape.allIndices_nodup [(Host.layout g.arguments).BM,(Host.layout g.arguments).D]
  rw [equal,List.nodup_append,List.nodup_cons] at distinct
  unfold writeTile
  rw [equal,List.foldl_append,List.foldl_cons,if_pos hi]
  rw [State.masked_scatter_frame]
  · exact State.write_same _ _ _ _
  · right
    intro j hj active same
    have ij := (unique_writer g.arguments r extent (metadata r extent xs) valid idx idx i j hi active same).2
    exact distinct.2.1.1 (ij ▸ hj)

#axiomsClean invocation_correct
#axiomsClean writeTile_output
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.KernelIO
