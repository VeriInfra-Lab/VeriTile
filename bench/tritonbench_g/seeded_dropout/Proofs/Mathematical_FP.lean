import bench.tritonbench_g.seeded_dropout.Proofs.Mathematical_Kernels
import VeriTile.Triton.Float.Structural
namespace VeriTile.Bench.TritonBenchG.SeededDropout.FPProof
open VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 3000

theorem prefix_run (storage : Storage) (kind : Random.IntegerType) (x y : RegionName)
    (N : Nat) (p : ℝ) (seed : SeedCarrier kind) (B : Nat) (hB : SourceDomain.LegalBlock B)
    {α : Type} [Inhabited α] (M : FP.Structural.Algebra α) (u : FP.Structural.State α) :
    FP.Structural.run M ((source storage kind Bool.false x y N p seed B hB).surfaceBody.take 4) u =
      FP.Structural.run M ((source storage kind Bool.true x y N p seed B hB).surfaceBody.take 4) u := by
  cases storage <;> cases kind
  all_goals simp [source, ComputeKernel.surfaceBody, FP.Structural.run, FP.Structural.step,
    FP.Structural.evalExpr, FP.Structural.evalOp_unfold, FP.Structural.numeric]
  all_goals
    congr 1
    funext i
    exact Nat.mul_comm _ _

theorem suffix (storage : Storage) (kind : Random.IntegerType) (x y : RegionName)
    (N : Nat) (p : ℝ) (seed : SeedCarrier kind) (B : Nat) (hB : SourceDomain.LegalBlock B) :
    (source storage kind Bool.false x y N p seed B hB).surfaceBody.drop 4 =
      (source storage kind Bool.true x y N p seed B hB).surfaceBody.drop 4 := by
  cases storage <;> cases kind <;> rfl

/-- Complete typed-source execution equality for every backend algebra and
initial state, including failure and all state fields. No numerical law,
finite-value premise, threshold restriction, or seed-validity premise is used.
The identical suffix retains canonical Philox, both RNG fp32 rounding stages,
the comparison, eager divide, and final storage conversion. -/
theorem whole_state (storage : Storage) (kind : Random.IntegerType) (x y : RegionName)
    (N : Nat) (p : ℝ) (seed : SeedCarrier kind) (B : Nat) (hB : SourceDomain.LegalBlock B)
    {α : Type} [Inhabited α] (M : FP.Structural.Algebra α) (u : FP.Structural.State α) :
    FP.Structural.exec M (source storage kind Bool.false x y N p seed B hB) u =
      FP.Structural.exec M (source storage kind Bool.true x y N p seed B hB) u := by
  change FP.Structural.run M (source storage kind Bool.false x y N p seed B hB).surfaceBody u =
    FP.Structural.run M (source storage kind Bool.true x y N p seed B hB).surfaceBody u
  conv_lhs => rw [← List.take_append_drop 4 (source storage kind Bool.false x y N p seed B hB).surfaceBody]
  conv_rhs => rw [← List.take_append_drop 4 (source storage kind Bool.true x y N p seed B hB).surfaceBody]
  simp only [FP.Structural.run_append, prefix_run, suffix]

end
end VeriTile.Bench.TritonBenchG.SeededDropout.FPProof
