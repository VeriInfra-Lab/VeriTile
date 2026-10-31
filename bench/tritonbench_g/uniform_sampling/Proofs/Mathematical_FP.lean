import bench.tritonbench_g.uniform_sampling.Proofs.Mathematical_Kernels
import VeriTile.Triton.Float.Structural

/-! Complete typed source/candidate equivalence. Only the integer lane-index
multiplication changes. The complete identical suffix keeps the modular
counter addition, all four Philox fields, fp32 uniform and affine operations,
and all four destination conversions/stores. This is structural execution
equality, not a numerical refinement from unfused arithmetic to native FMA.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.FPProof
open VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

theorem run_insert {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (before after : List ComputeStmt) (a b : ComputeStmt)
    (h : ∀ s, FP.Structural.step M a s = FP.Structural.step M b s)
    (s : FP.Structural.State α) :
    FP.Structural.run M (before ++ a :: after) s =
      FP.Structural.run M (before ++ b :: after) s := by
  rw [FP.Structural.run_append, FP.Structural.run_append]
  cases hp : FP.Structural.run M before s <;>
    simp only [Option.bind_none, Option.bind_some, FP.Structural.run, h]

end
end VeriTile.Bench.TritonBenchG.UniformSampling.FPProof
