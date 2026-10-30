import bench.tritonbench_g.triton_matmul.IO
import bench.tritonbench_g.triton_matmul.Proofs.PublicIO

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly.PublicIO
open VeriTile.Triton FP.Structural
noncomputable section

theorem logical_iff (g : Geometry) (ids : List Nat)
    (f : (Fin g.rows → Fin g.inner → ℝ) → (Fin g.inner → Fin g.columns → ℝ) →
      Fin g.rows → Fin g.columns → ℝ) :
    (logicalRealIO g ids).Implements (fun xs => f xs.1 xs.2) ↔
      (io g KernelRealModel.ideal KernelRealModel.idealDot ids).Implements (expected g f) := by
  have eq : (logicalRealIO g ids).lift (fun xs => f xs.1 xs.2) = expected g f := by
    funext xs port i
    fin_cases port
    rfl
  change (io g KernelRealModel.ideal KernelRealModel.idealDot ids).Implements _ ↔ _
  rw [eq]
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly.PublicIO
