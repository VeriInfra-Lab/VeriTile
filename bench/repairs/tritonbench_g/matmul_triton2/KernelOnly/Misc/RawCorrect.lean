import bench.repairs.tritonbench_g.matmul_triton2.revision001.Proofs.Grid
import bench.repairs.tritonbench_g.matmul_triton2.revision001.Proofs.Real
namespace VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.RawKernel
open VeriTile.Triton FP.Structural Revision001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section

/-- Complete source grid, independent dense/strided matrix mathematics and
full raw memory frame. Geometry contains only kernel scalars and addresses. -/
theorem kernel_real (rest : Algebra ℝ) (p : Core.Parameters)
    (geometry : Geometry.Valid (Core.geometry p)) (inner : 0 < p.blockInner)
    (input : p.input ≠ p.output) (weight : p.weight ≠ p.output) (before : State ℝ) :
    ∃ after, Core.grid rest Core.realPrimitives p before=some after ∧
      (∀ x : Core.LogicalIndex p,after.mem p.output (Geometry.address (Core.geometry p) x)=
        .mk .real (Core.realMatmul p before x.1.val x.2.val)) ∧
      (∀ r i,(r≠p.output ∨ ∀ x : Core.LogicalIndex p,i≠Geometry.address (Core.geometry p) x) →
        after.mem r i=before.mem r i) := by
  obtain ⟨after,run,values,frame⟩ := Core.grid_correct rest Core.realPrimitives p geometry inner input weight before
  refine ⟨after,run,?_,?_⟩
  · intro x
    rw [values x trivial]
    change Cell.mk .real (Core.logical Core.realPrimitives p before x.1.val x.2.val)=_
    rw [Core.logical_realMatmul p before x.1.val x.2.val inner]
  · intro r i outside
    apply frame r i
    exact outside.imp id (fun h x _ => h x)

/-- A raw grid with no programs preserves the complete state. -/
theorem kernel_empty {α : Type} [Inhabited α] (rest : Algebra α) (ops : Core.Primitives α)
    (p : Core.Parameters) (before : State α) (empty : Geometry.count (Core.geometry p)=0) :
    Core.grid rest ops p before=some before := by
  simp [Core.grid,empty,GridExecution.sequence]

#axiomsClean kernel_real
#axiomsClean kernel_empty
#specNonCircular Core.realMatmul avoiding [Core.grid,Core.source,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.RawKernel
