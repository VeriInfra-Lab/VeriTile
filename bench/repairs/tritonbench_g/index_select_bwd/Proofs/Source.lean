import bench.repairs.tritonbench_g.index_select_bwd.Kernels
import bench.tritonbench_g.vector_addition.Proofs.TypedFloat

namespace VeriTile.Bench.Repairs.IndexSelectBwd.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- FP32 loads and identity casts retain their raw payload. This add-only source uses only FP32 addition. The contract excludes lazy view flags. -/
def backend {α : Type} (rest : Algebra α) : Algebra α :=
  {rest with fp32Load := id, cast := fun _ _ _ x => x}

def scalar {α : Type} (M : Algebra α) (x y : α) : α :=
  M.binary (some .fp32) .real .add x y

def expected {α : Type} [Inhabited α] (M : Algebra α) (p : Launch) (s : State α) :=
  (TileShape.allIndices [p.block]).foldl (fun mem i =>
    if i.1.val<p.count then fun r o =>
      if r=p.destination ∧ o=i.1.val*p.stride then
        Cell.mk .real (scalar M
          ((s.mem p.destination (i.1.val*p.stride)).read .real)
          ((s.mem p.incoming i.1.val).read .real))
      else mem r o
    else mem) s.mem

/-- Constructed complete typed execution, not an assumed launch result. -/
def execution {α : Type} [Inhabited α] (M : Algebra α) (p : Launch) (s : State α) :
    {t : State α // FP.Structural.exec (backend M) (kernel p) s=some t} := by
  rcases p with ⟨dst,inp,count,stride,block⟩
  refine ⟨?_,?_⟩
  swap
  simp [kernel,FP.Structural.exec,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,
    Option.bind,backend,numericLt,natLt,bop,numeric,store,Region.cast,
    ComputeDType.eraseDType,ofFloat,toFloat]
  rfl

theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α) (p : Launch) (s : State α) :
    (execution M p s).val.mem=expected M p s := by
  rcases p with ⟨dst,inp,count,stride,block⟩
  simp [execution,expected,scalar,bop,Broadcast.leftIndex,Broadcast.rightIndex,
    masked_scatter_memory,State.setReg,backend]
  congr 1
  funext mem i
  by_cases h : i.1.val<count
  · simp [h,Int.ofNat_lt,Int.toNat_mul]
  · simp [h,Int.ofNat_lt]

theorem expected_scatter {α : Type} [Inhabited α] (M : Algebra α) (p : Launch) (s : State α) :
    expected M p s = ((TileShape.allIndices [p.block]).foldl (fun state i =>
      if i.1.val<p.count then state.write p.destination (i.1.val*p.stride)
        (Cell.mk .real (scalar M ((s.mem p.destination (i.1.val*p.stride)).read .real)
          ((s.mem p.incoming i.1.val).read .real))) else state) s).mem := by
  rw [masked_scatter_memory]
  rfl

theorem readback {α : Type} [Inhabited α] (M : Algebra α) (p : Launch) (s : State α)
    (unitStride : p.stride=1) (i : TileIndex [p.block]) (active : i.1.val<p.count) :
    (execution M p s).val.mem p.destination i.1.val =
      Cell.mk .real (scalar M ((s.mem p.destination i.1.val).read .real)
        ((s.mem p.incoming i.1.val).read .real)) := by
  rw [execution_memory,expected_scatter]
  have injective : Function.Injective (fun i : TileIndex [p.block] => i.1.val*p.stride) := by
    intro a b eq
    simp only [unitStride,Nat.mul_one] at eq
    exact Prod.ext (Fin.ext eq) (Subsingleton.elim _ _)
  simpa only [unitStride,Nat.mul_one] using State.masked_scatter_readback p.destination
    (fun i : TileIndex [p.block] => i.1.val*p.stride)
    (fun i => Cell.mk .real (scalar M ((s.mem p.destination (i.1.val*p.stride)).read .real)
      ((s.mem p.incoming i.1.val).read .real)))
    (fun i => i.1.val<p.count) s i active injective

theorem frame {α : Type} [Inhabited α] (M : Algebra α) (p : Launch) (s : State α)
    (r : RegionName) (o : Nat)
    (miss : r≠p.destination ∨ ∀ i : TileIndex [p.block],i.1.val<p.count → o≠i.1.val*p.stride) :
    (execution M p s).val.mem r o=s.mem r o := by
  rw [execution_memory,expected_scatter]
  exact State.masked_scatter_frame _ _ _ _ _ _ _
    (miss.imp id (fun h i _ => h i)) s

#axiomsClean readback
#axiomsClean frame
#axiomsClean execution
#axiomsClean execution_memory
end
end VeriTile.Bench.Repairs.IndexSelectBwd.Source
