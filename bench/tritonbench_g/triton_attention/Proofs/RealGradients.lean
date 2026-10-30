import bench.tritonbench_g.triton_attention.Proofs.RealMath
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Mathematical
open scoped BigOperators
open VeriTile.Bench.Repairs.TritonAttention.Whole001.Math
noncomputable section
set_option maxHeartbeats 4000000
variable {N D : Nat}

/-- Any finite shift gives the same normalized causal distribution. This
lets the execution proof use its computed maximum without assuming a result. -/
def shiftedNormalizer (Q K : Tensor N D) (scale : ℝ) (M : Fin N → ℝ) (i : Fin N) : ℝ :=
  ∑ j,sourceP Q K scale M i j

theorem shiftedNormalizer_pos (Q K : Tensor N D) (scale : ℝ) (M : Fin N → ℝ) (i : Fin N) :
    0<shiftedNormalizer Q K scale M i := by
  rw [shiftedNormalizer,shifted_partition]
  exact div_pos (partition_pos Q K scale i) (Real.exp_pos _)

theorem arbitraryShift_probability (Q K : Tensor N D) (scale : ℝ) (M : Fin N → ℝ) (i j : Fin N) :
    sourceP Q K scale M i j/shiftedNormalizer Q K scale M i=probability Q K scale i j := by
  rw [sourceP_eq_weight_div,shiftedNormalizer,shifted_partition,probability]
  field_simp

theorem arbitraryShift_delta (Q K V G : Tensor N D) (scale : ℝ) (M : Fin N → ℝ) (i : Fin N) :
    preprocessDelta (VeriTile.Bench.Repairs.TritonAttention.Whole001.Math.attention Q K V scale) G
      (shiftedNormalizer Q K scale M) i=correction Q K V G scale i/shiftedNormalizer Q K scale M i := by
  rw [correction_eq_output_dot]
  simp only [preprocessDelta,preprocessDO,Finset.sum_div,mul_div_assoc]

theorem arbitraryShift_scoreGradient (Q K V G : Tensor N D) (scale : ℝ) (M : Fin N → ℝ) (i j : Fin N) :
    sourceDS Q K V G (VeriTile.Bench.Repairs.TritonAttention.Whole001.Math.attention Q K V scale)
      scale (shiftedNormalizer Q K scale M) M i j=scoreGradient Q K V G scale i j*scale := by
  rw [sourceDS,preprocess_dot,arbitraryShift_delta,←sub_div,←mul_div_assoc,
    mul_div_right_comm,arbitraryShift_probability]
  rfl

theorem arbitraryShift_gradients (Q K V G : Tensor N D) (scale : ℝ) (M : Fin N → ℝ) :
    sourceDQ Q K V G (VeriTile.Bench.Repairs.TritonAttention.Whole001.Math.attention Q K V scale)
      scale (shiftedNormalizer Q K scale M) M=gradientQ Q K V G scale ∧
    sourceDK Q K V G (VeriTile.Bench.Repairs.TritonAttention.Whole001.Math.attention Q K V scale)
      scale (shiftedNormalizer Q K scale M) M=gradientK Q K V G scale ∧
    sourceDV Q K G scale (shiftedNormalizer Q K scale M) M=gradientV Q K G scale := by
  constructor
  · funext i e
    simp only [sourceDQ,gradientQ,arbitraryShift_scoreGradient]
  constructor
  · funext j e
    simp only [sourceDK,gradientK,arbitraryShift_scoreGradient]
  · funext j e
    simp only [sourceDV,gradientV,preprocessDO]
    apply Finset.sum_congr rfl
    intro i _
    rw [←mul_div_assoc,mul_div_right_comm,arbitraryShift_probability]

#axiomsClean arbitraryShift_gradients
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Mathematical
