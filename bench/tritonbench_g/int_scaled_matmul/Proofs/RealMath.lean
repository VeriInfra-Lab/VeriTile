import bench.tritonbench_g.int_scaled_matmul.Proofs.RealIO

namespace VeriTile.Bench.Repairs.IntScaledMatmul.RealScaled
noncomputable section
/-- Mathlib characterizes the independent toward-zero scalar before the
explicit final signed32 projection. -/
theorem truncate_spec (x : ℝ) :
    (0≤x → (truncate x : ℝ)≤x ∧ x<(truncate x : ℝ)+1) ∧
    (x<0 → (truncate x : ℝ)-1<x ∧ x≤(truncate x : ℝ)) := by
  constructor
  · intro positive
    rw [truncate,if_neg (not_lt.mpr positive)]
    exact ⟨Int.floor_le x,Int.lt_floor_add_one x⟩
  · intro negative
    rw [truncate,if_pos negative]
    have upper := Int.ceil_lt_add_one x
    exact ⟨by linarith,Int.le_ceil x⟩
#axiomsClean truncate_spec
end
end VeriTile.Bench.Repairs.IntScaledMatmul.RealScaled
