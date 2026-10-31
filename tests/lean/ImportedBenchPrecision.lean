import bench.bug_detection.tasks.rmsnorm_w_fp16_roundtrip.Kernels
import bench.bug_detection.tasks.gemma_one_plus_w_bf16.Kernels
import VeriTile.Triton.Source
import VeriTile.Meta.StatementAudit
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Tactic

/-! Source-graph regressions for the declared precision mutations. These
checks do not solve either challenge or establish native backend equivalence. -/
namespace VeriTile.Tests.ImportedBenchPrecision
open VeriTile.Triton
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000

private def assignment (k : ComputeKernel) (name : RegName) (occurrence : Nat := 0) :=
  (k.surfaceBody.drop (Source.prefixBefore k.surfaceBody name occurrence).length).head?

private abbrev Rms := Bench.BugDetection.rmsnorm_w_fp16_roundtrip.Kernels.original
private abbrev RmsCandidate := Bench.BugDetection.rmsnorm_w_fp16_roundtrip.Kernels.candidate
private abbrev Gemma := Bench.BugDetection.gemma_one_plus_w_bf16.Kernels.original
private abbrev GemmaCandidate := Bench.BugDetection.gemma_one_plus_w_bf16.Kernels.candidate

/-- The complete RMS program changes only by inserting the weight roundtrip.
In particular, its square, sum, epsilon, rsqrt and output are retained. -/
theorem rms_changes_only_weight :
    RmsCandidate.surfaceBody =
      Source.prefixBefore Rms.surfaceBody "y" ++
      [.assign .real [2048] "w" (.compute (.alg .fp32
        (.castFloat .fp16 .real (.castFloat .real .fp16 (.ref .real [2048] "w")))))] ++
      Rms.surfaceBody.drop (Source.prefixBefore Rms.surfaceBody "y").length := rfl

theorem gemma_preserves_normalization :
    Source.prefixBefore GemmaCandidate.surfaceBody "z" 1 =
      Source.prefixBefore Gemma.surfaceBody "scale" := rfl

/-- Addition executes in FP32 before the scale is narrowed to BF16. -/
theorem gemma_rounds_the_sum :
    assignment GemmaCandidate "scale" = some
      (.assign .bf16 [2048] "scale" (.alg
        (.castFloat .real .bf16 (.withPrecision .real .fp32
          (.add .real .scalarL (.const 1.0) (.ref .real [2048] "w")))))) := rfl

theorem gemma_does_not_round_weight_first :
    assignment GemmaCandidate "w" 1 = none := rfl

/-- A finite mask has nonzero probability in the exact Real specification;
different finite masks cannot be identified just because FP underflows. -/
theorem finite_mask_probability_distinguishes {a b : ℝ} (hab : a ≠ b) :
    Real.exp a / (1 + 127 * Real.exp a) ≠
      Real.exp b / (1 + 127 * Real.exp b) := by
  have ha : 1 + 127 * Real.exp a ≠ 0 := ne_of_gt (by positivity)
  have hb : 1 + 127 * Real.exp b ≠ 0 := ne_of_gt (by positivity)
  intro h
  have hc := (div_eq_div_iff ha hb).mp h
  apply hab
  apply Real.exp_injective
  nlinarith

#axiomsClean rms_changes_only_weight
#axiomsClean gemma_preserves_normalization
#axiomsClean gemma_rounds_the_sum
#axiomsClean gemma_does_not_round_weight_first
#axiomsClean finite_mask_probability_distinguishes
end VeriTile.Tests.ImportedBenchPrecision
