import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.SourceBinding
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Contract
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.InputBinding

namespace VeriTile.Bench.Repairs.LayerNormWelfold.KernelBoundary
open VeriTile.Triton FP.Structural Reuse Samples
noncomputable section

private theorem rows_binding {α : Type} [Inhabited α] (M : Algebra α)
    (kernel : ComputeKernel) (rows : List Nat) (s : State α) :
    Source.runRows M kernel rows s = Reuse.runRows M kernel rows s := by
  induction rows generalizing s with
  | nil => rfl
  | cons row rows ih => simp only [Source.runRows, Reuse.runRows, ih, atRow]

/-- Successful execution of the canonical typed source, every active store,
and all memory outside the three output footprints. -/
theorem complete {α : Type} [Inhabited α] (M : Algebra α) (eq32 : α → α → Bool)
    (s : State α) (Y Mean Rstd : RegionName) (S D : Nat) (c : Configuration)
    (sep : Separation "X" "W" "Bias" Y Mean Rstd) :
    ∃ final, Source.launch (withEquality M eq32) s Y Mean Rstd S D c.block (SourceBinding.tree c.tree) = some final ∧
      GridEffect M s final "X" "W" "Bias" Y Mean Rstd D c.block c.tree (List.range S) := by
  obtain ⟨final, run, effect⟩ := whole_grid M eq32 s "X" "W" "Bias" Y Mean Rstd S D
    c.block c.tree c.positive sep (List.range S)
  refine ⟨final, ?_, effect⟩
  simpa only [Source.launch, rows_binding, SourceBinding.row_binding] using run

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

#axiomsClean complete
#axiomsClean mean_logical
#axiomsClean inverse_logical
end
end VeriTile.Bench.Repairs.LayerNormWelfold.KernelBoundary
