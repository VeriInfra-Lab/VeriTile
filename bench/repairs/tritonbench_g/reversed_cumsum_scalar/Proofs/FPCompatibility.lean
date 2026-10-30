import bench.repairs.tritonbench_g.reversed_cumsum_scalar.FPPrograms
import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Kernels
import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.Rewrite
import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.FPExecution

/-! The intended input-index commutation on the complete repaired kernel.
All floating operations and their precision remain in their original context. -/
namespace VeriTile.Bench.Repairs.ReversedCumsumScalar.Equivalence
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec
noncomputable section

/-- This is equality between the displayed source pair, with no dtype erasure. -/
theorem original_source (h : Call) (c : Config) : (original h c).kernel=Kernels.source h c := rfl

/-- Both complete kernels succeed, on any admitted row and total scan backend.
No successful-execution or output-equality premise is supplied. -/
theorem successful_pair {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (scan : FPValues.ScanBackend α) (h : Call) (layout : Geometry.OutputLayout h)
    (c : Config) (chosen : c∈configurations) (u : FP.Structural.State α) (row : u.pids 0<h.B*h.H) :
    ∃ v, FP.Structural.exec (FPValues.machine M scan) (original h c).kernel u=some v ∧
      FP.Structural.exec (FPValues.machine M scan) (indexCommuted h c).kernel u=some v ∧
      FPExecution.Post M scan h c (Kernels.configurations_positive c chosen).1 u v := by
  obtain ⟨v,executed,post⟩ := FPExecution.source_correct M scan h layout c
    (Kernels.configurations_positive c chosen).1 u row
  exact ⟨v,executed,(Rewrite.whole_state _ h c u).symm.trans executed,post⟩

#axiomsClean original_source
#axiomsClean successful_pair
end
end VeriTile.Bench.Repairs.ReversedCumsumScalar.Equivalence
