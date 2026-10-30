import bench.repairs.tritonbench_g.layer_norm_welfold.Contract

namespace VeriTile.Bench.Repairs.LayerNormWelfold.Math
noncomputable section

theorem variance_nonnegative (x : Nat → ℝ) (D row : Nat) : 0 ≤ variance x D row := by
  apply div_nonneg
  · apply List.sum_nonneg
    intro term member
    obtain ⟨col, _, rfl⟩ := List.mem_map.mp member
    exact sq_nonneg _
  · exact Nat.cast_nonneg D

theorem inverse_positive (x : Nat → ℝ) (D row : Nat) : 0 < inverse x D row := by
  apply one_div_pos.mpr
  apply Real.sqrt_pos.mpr
  have h := variance_nonnegative x D row
  norm_num at ⊢
  linarith

/-- Zero-filled padded loads alone still corrupt the unmasked variance pass:
each of B-D padded lanes contributes mean². -/
theorem padded_square_nonzero : ((0 : ℝ) - 2)^2 = 4 := by norm_num

#axiomsClean variance_nonnegative
#axiomsClean inverse_positive
#axiomsClean padded_square_nonzero
end
end VeriTile.Bench.Repairs.LayerNormWelfold.Math
