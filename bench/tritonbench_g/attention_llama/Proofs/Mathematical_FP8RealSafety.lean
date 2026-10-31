import bench.tritonbench_g.attention_llama.Proofs.Mathematical_FP8RealSafetyLoop

/-! Address safety of both complete byte-input sources. Bounds are derived
from the actual initialization, counter, input pointer advances and final
masked store, independently of the real attention formula. -/
namespace VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealSafety
open VeriTile.Triton RealKernel FP8RealLoop FP8RealLoopFrame FP8RealSafetyLoop
set_option maxRecDepth 16000
set_option maxHeartbeats 1000000

section
variable (Q K V Out : RegionName) (scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD : Nat) (causal candidate : Bool)

end

/-- A whole-block span reduces the scheduled address requirements to ordinary
row bounds. N may be larger than the causal span. -/
theorem whole_blocks (bounds : RegionBounds) (R : RegionName)
    (base sr sc N BN BD T : Nat)
    (hb : ∀ r, r < BN * T → ∀ d : Fin BD,
      base + r * sr + d.val * sc < bounds R) :
    AccessesFrom bounds R base sr sc 0 (BN * T) N BN BD := by
  intro k hk j d _
  simp only [Nat.zero_add] at hk ⊢
  apply hb _ ?_ d
  have hkt : k < T := by nlinarith
  have hs := Nat.mul_le_mul_right BN (Nat.succ_le_of_lt hkt)
  have hj := j.isLt
  nlinarith

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealSafety
