import bench.tritonbench_g.layer_norm_welfold.Proofs.SourceBinding
import bench.tritonbench_g.layer_norm_welfold.Proofs.Contract
import bench.tritonbench_g.layer_norm_welfold.Proofs.InputBinding

namespace VeriTile.Bench.Repairs.LayerNormWelfold.KernelBoundary
open VeriTile.Triton FP.Structural Reuse Samples
noncomputable section

theorem mean_logical (s : State ℝ) (x : Nat → ℝ) (S D row : Nat) (hr : row < S)
    (h : ∀ r, r < S → ∀ col, col < D → (s.mem "X" (col + D * r)).read .real = x (col + D * r)) :
    targetMean (atRow s row) "X" D = Math.mean x D row := by
  unfold targetMean Math.mean
  congr 1
  apply congrArg List.sum
  apply List.map_congr_left
  intro col hc
  exact h row hr col (List.mem_range.mp hc)

theorem inverse_logical (s : State ℝ) (x : Nat → ℝ) (S D row : Nat) (hr : row < S)
    (h : ∀ r, r < S → ∀ col, col < D → (s.mem "X" (col + D * r)).read .real = x (col + D * r)) :
    targetInverse (atRow s row) "X" D = Math.inverse x D row := by
  have hm := mean_logical s x S D row hr h
  have hv : targetVariance (atRow s row) "X" D = Math.variance x D row := by
    unfold targetVariance Math.variance
    rw [hm]
    congr 1
    apply congrArg List.sum
    apply List.map_congr_left
    intro col hc
    rw [show ((atRow s row).mem "X" (col + D * (atRow s row).pids 0)).read .real =
      x (col + D * row) from h row hr col (List.mem_range.mp hc)]
  simp only [targetInverse, Math.inverse, hv]

#axiomsClean mean_logical
#axiomsClean inverse_logical
end
end VeriTile.Bench.Repairs.LayerNormWelfold.KernelBoundary
