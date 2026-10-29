import bench.repairs.tritonbench_g.rms_norm_triton.Proofs.ActualRow
import bench.tritonbench_g.rms_norm_triton.Proofs.WholeMath
namespace VeriTile.Bench.Repairs.RmsNormTriton.RuntimeReal
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.RmsNormTriton Whole
noncomputable section
/-- Only the Real interpretation identifies the two exact real embeddings.
No corresponding identity is assumed for an opaque floating algebra. -/
theorem rowValue_argument (c : Layout) (eps : ℝ) (xs ws : Nat → ℝ) (i : Nat) :
    ActualRow.rowValue realAlgebra c (ActualRow.argument realAlgebra eps) xs ws i =
      Whole.rowValue realAlgebra c eps xs ws i := rfl

theorem rowValue_real (c : Layout) (eps : ℝ) (xs ws : Nat → ℝ) (i : Nat) :
    ActualRow.rowValue realAlgebra c (ActualRow.argument realAlgebra eps) xs ws i =
      realFormula c.N eps xs ws i := by
  rw [rowValue_argument]
  exact Whole.rowValue_real c eps xs ws i
#axiomsClean rowValue_argument
#axiomsClean rowValue_real
end
end VeriTile.Bench.Repairs.RmsNormTriton.RuntimeReal
