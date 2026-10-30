import bench.tritonbench_g.swiglu_fwd.IO
import bench.tritonbench_g.swiglu_fwd.Proofs.PublicIO
namespace VeriTile.Bench.Repairs.SwigluFwd.Logical
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluFwd.Whole
noncomputable section
/-- Equivalence of the complete old source judgment, including its full masked B window. -/
theorem judgment_iff (c : Layout) (B row tile : Nat) (f : ℝ → ℝ → ℝ) :
    (io c B row tile).Implements (fun xs i => f (xs.x i) (xs.y i)) ↔
    (KernelPublic.io realAlgebra id id c B row tile).Implements
      (KernelPublic.target realAlgebra id id c B row tile f) := by
  have same : (io c B row tile).lift (fun xs i => f (xs.x i) (xs.y i)) =
      KernelPublic.target realAlgebra id id c B row tile f := by
    funext xs port
    fin_cases port
    rfl
  change (KernelPublic.io realAlgebra id id c B row tile).Implements _ ↔ _
  rw [same]
theorem reader_implements (c : Layout) (B row tile : Nat) :
    (io c B row tile).Implements (fun xs i => xs.x i * xs.y i / (1 + Real.exp (-xs.x i))) := by
  apply (judgment_iff c B row tile (fun x y : ℝ => x * y / (1 + Real.exp (-x)))).2
  exact KernelPublic.implements realAlgebra id id
    (fun x y : ℝ => x * y / (1 + Real.exp (-x)))
    (by intro x y
        simpa only [id_eq,VeriTile.Bench.TritonBenchG.SwigluFwd.Math.forward] using scalar_real x y)
    c B row tile

end
end VeriTile.Bench.Repairs.SwigluFwd.Logical
