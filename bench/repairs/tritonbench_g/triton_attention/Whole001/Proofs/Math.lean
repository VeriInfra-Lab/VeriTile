import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Data.Finset.Max
import Mathlib.Tactic
import VeriTile.Meta.StatementAudit

/-!
Independent Real mathematics for causal attention and the source backward
formulas.  No kernel execution, floating-point operation, or numerical oracle
occurs in the mathematical specification.  The scale is arbitrary; in
particular every positive scale and every positive sequence/head dimension is
covered.  Real cast erasure is a separate semantic interpretation, never an
assertion that floating-point arithmetic equals Real arithmetic.
-/

namespace VeriTile.Bench.Repairs.TritonAttention.Whole001.Math

open scoped BigOperators

abbrev Tensor (N D : Nat) := Fin N → Fin D → ℝ

variable {N D : Nat}

noncomputable def score (Q K : Tensor N D) (scale : ℝ) (i j : Fin N) : ℝ :=
  (∑ e, Q i e * K j e) * scale

noncomputable def causalWeight (Q K : Tensor N D) (scale : ℝ) (i j : Fin N) : ℝ :=
  if j ≤ i then Real.exp (score Q K scale i j) else 0

noncomputable def partition (Q K : Tensor N D) (scale : ℝ) (i : Fin N) : ℝ :=
  ∑ j, causalWeight Q K scale i j

@[standalone_spec] noncomputable def probability (Q K : Tensor N D) (scale : ℝ)
    (i j : Fin N) : ℝ :=
  causalWeight Q K scale i j / partition Q K scale i

@[standalone_spec] noncomputable def attention (Q K V : Tensor N D) (scale : ℝ) :
    Tensor N D := fun i e => ∑ j, probability Q K scale i j * V j e

noncomputable def dotGV (V G : Tensor N D) (i j : Fin N) : ℝ :=
  ∑ e, G i e * V j e

noncomputable def correction (Q K V G : Tensor N D) (scale : ℝ) (i : Fin N) : ℝ :=
  ∑ j, probability Q K scale i j * dotGV V G i j

/-- Cotangent of the scaled score, using the ordinary softmax Jacobian. -/
@[standalone_spec] noncomputable def scoreGradient (Q K V G : Tensor N D) (scale : ℝ)
    (i j : Fin N) : ℝ :=
  probability Q K scale i j * (dotGV V G i j - correction Q K V G scale i)

@[standalone_spec] noncomputable def gradientQ (Q K V G : Tensor N D) (scale : ℝ) :
    Tensor N D := fun i e => ∑ j, (scoreGradient Q K V G scale i j * scale) * K j e

@[standalone_spec] noncomputable def gradientK (Q K V G : Tensor N D) (scale : ℝ) :
    Tensor N D := fun j e => ∑ i, (scoreGradient Q K V G scale i j * scale) * Q i e

@[standalone_spec] noncomputable def gradientV (Q K : Tensor N D) (G : Tensor N D)
    (scale : ℝ) : Tensor N D := fun j e => ∑ i, probability Q K scale i j * G i e

def causalSet (i : Fin N) : Finset (Fin N) := Finset.univ.filter (· ≤ i)

theorem causalSet_nonempty (i : Fin N) : (causalSet i).Nonempty := by
  exact ⟨i, by simp [causalSet]⟩

/-- The saved maximum is derived from Q/K, rather than supplied as an output premise. -/
@[standalone_spec] noncomputable def savedMax (Q K : Tensor N D) (scale : ℝ)
    (i : Fin N) : ℝ :=
  (causalSet i).sup' (causalSet_nonempty i) (score Q K scale i)

/-- The source reconstructs an unnormalized, max-shifted exponential. -/
noncomputable def sourceP (Q K : Tensor N D) (scale : ℝ) (M : Fin N → ℝ)
    (i j : Fin N) : ℝ :=
  if j ≤ i then Real.exp (score Q K scale i j - M i) else 0

@[standalone_spec] noncomputable def savedNormalizer (Q K : Tensor N D) (scale : ℝ)
    (i : Fin N) : ℝ := ∑ j, sourceP Q K scale (savedMax Q K scale) i j

@[standalone_spec] noncomputable def savedOutput (Q K V : Tensor N D) (scale : ℝ) :
    Tensor N D := fun i e =>
  (∑ j, sourceP Q K scale (savedMax Q K scale) i j * V j e) /
    savedNormalizer Q K scale i

noncomputable def preprocessDO (G : Tensor N D) (L : Fin N → ℝ) : Tensor N D :=
  fun i e => G i e / L i

noncomputable def preprocessDelta (Y G : Tensor N D) (L : Fin N → ℝ)
    (i : Fin N) : ℝ := ∑ e, Y i e * preprocessDO G L i e

noncomputable def sourceDS (Q K V G Y : Tensor N D) (scale : ℝ)
    (L M : Fin N → ℝ) (i j : Fin N) : ℝ :=
  sourceP Q K scale M i j *
    ((∑ e, preprocessDO G L i e * V j e) - preprocessDelta Y G L i) * scale

noncomputable def sourceDQ (Q K V G Y : Tensor N D) (scale : ℝ)
    (L M : Fin N → ℝ) : Tensor N D :=
  fun i e => ∑ j, sourceDS Q K V G Y scale L M i j * K j e

noncomputable def sourceDK (Q K V G Y : Tensor N D) (scale : ℝ)
    (L M : Fin N → ℝ) : Tensor N D :=
  fun j e => ∑ i, sourceDS Q K V G Y scale L M i j * Q i e

noncomputable def sourceDV (Q K G : Tensor N D) (scale : ℝ)
    (L M : Fin N → ℝ) : Tensor N D :=
  fun j e => ∑ i, sourceP Q K scale M i j * preprocessDO G L i e

theorem causalWeight_nonneg (Q K : Tensor N D) (scale : ℝ) (i j : Fin N) :
    0 ≤ causalWeight Q K scale i j := by
  unfold causalWeight
  split_ifs <;> positivity

theorem partition_pos (Q K : Tensor N D) (scale : ℝ) (i : Fin N) :
    0 < partition Q K scale i := by
  apply Finset.sum_pos' (fun j _ => causalWeight_nonneg Q K scale i j)
  exact ⟨i, Finset.mem_univ _, by simp [causalWeight, Real.exp_pos]⟩

theorem probability_sum (Q K : Tensor N D) (scale : ℝ) (i : Fin N) :
    ∑ j, probability Q K scale i j = 1 := by
  simp only [probability, ← Finset.sum_div]
  exact div_self (ne_of_gt (partition_pos Q K scale i))

theorem savedMax_ge (Q K : Tensor N D) (scale : ℝ) (i j : Fin N) (h : j ≤ i) :
    score Q K scale i j ≤ savedMax Q K scale i := by
  exact Finset.le_sup' _ (by simp [causalSet, h])

theorem savedMax_attained (Q K : Tensor N D) (scale : ℝ) (i : Fin N) :
    ∃ j : Fin N, j ≤ i ∧ savedMax Q K scale i = score Q K scale i j := by
  obtain ⟨j, hj, heq⟩ := Finset.exists_mem_eq_sup' (causalSet_nonempty i) (score Q K scale i)
  exact ⟨j, (Finset.mem_filter.mp hj).2, heq⟩

theorem sourceP_eq_weight_div (Q K : Tensor N D) (scale : ℝ) (M : Fin N → ℝ)
    (i j : Fin N) :
    sourceP Q K scale M i j = causalWeight Q K scale i j / Real.exp (M i) := by
  by_cases h : j ≤ i <;> simp [sourceP, causalWeight, h, Real.exp_sub]

theorem shifted_partition (Q K : Tensor N D) (scale : ℝ) (M : Fin N → ℝ)
    (i : Fin N) :
    (∑ j, sourceP Q K scale M i j) = partition Q K scale i / Real.exp (M i) := by
  simp only [sourceP_eq_weight_div, ← Finset.sum_div, partition]

theorem savedNormalizer_pos (Q K : Tensor N D) (scale : ℝ) (i : Fin N) :
    0 < savedNormalizer Q K scale i := by
  rw [savedNormalizer, shifted_partition]
  exact div_pos (partition_pos Q K scale i) (Real.exp_pos _)

theorem sourceP_div_normalizer (Q K : Tensor N D) (scale : ℝ)
    (i j : Fin N) :
    sourceP Q K scale (savedMax Q K scale) i j / savedNormalizer Q K scale i =
      probability Q K scale i j := by
  rw [sourceP_eq_weight_div, savedNormalizer, shifted_partition, probability]
  field_simp

/-- Stable forward normalization eliminates the derived row maximum. -/
theorem savedOutput_eq_attention (Q K V : Tensor N D) (scale : ℝ) :
    savedOutput Q K V scale = attention Q K V scale := by
  funext i e
  simp only [savedOutput, attention, Finset.sum_div]
  apply Finset.sum_congr rfl
  intro j _
  rw [mul_div_right_comm, sourceP_div_normalizer]

/-- The correction is also the inner product of the forward output and G. -/
theorem correction_eq_output_dot (Q K V G : Tensor N D) (scale : ℝ) (i : Fin N) :
    correction Q K V G scale i = ∑ e, attention Q K V scale i e * G i e := by
  simp only [correction, dotGV, attention, Finset.mul_sum, Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro e _
  apply Finset.sum_congr rfl
  intro j _
  ring

theorem preprocess_dot (V G : Tensor N D) (L : Fin N → ℝ) (i j : Fin N) :
    (∑ e, preprocessDO G L i e * V j e) = dotGV V G i j / L i := by
  simp only [preprocessDO, dotGV, Finset.sum_div]
  apply Finset.sum_congr rfl
  intro e _
  ring

theorem preprocessDelta_eq (Q K V G : Tensor N D) (scale : ℝ) (i : Fin N) :
    preprocessDelta (savedOutput Q K V scale) G (savedNormalizer Q K scale) i =
      correction Q K V G scale i / savedNormalizer Q K scale i := by
  rw [savedOutput_eq_attention, correction_eq_output_dot]
  simp only [preprocessDelta, preprocessDO, Finset.sum_div, mul_div_assoc]

/-- Exactly the source's `p * (dot(G/L,V) - Delta) * scale`. -/
theorem sourceDS_eq_scoreGradient (Q K V G : Tensor N D) (scale : ℝ) (i j : Fin N) :
    sourceDS Q K V G (savedOutput Q K V scale) scale
      (savedNormalizer Q K scale) (savedMax Q K scale) i j =
        scoreGradient Q K V G scale i j * scale := by
  rw [sourceDS, preprocess_dot, preprocessDelta_eq, ← sub_div, ← mul_div_assoc,
    mul_div_right_comm, sourceP_div_normalizer]
  rfl

/-- A conventional softmax Jacobian, independent of source preprocessing. -/
noncomputable def softmaxJacobian (Q K : Tensor N D) (scale : ℝ)
    (i j k : Fin N) : ℝ :=
  probability Q K scale i j * ((if j = k then 1 else 0) - probability Q K scale i k)

/-- The ideal score gradient is the transpose-Jacobian pullback of `G Vᵀ`. -/
theorem scoreGradient_eq_jacobian_pullback (Q K V G : Tensor N D) (scale : ℝ)
    (i j : Fin N) :
    scoreGradient Q K V G scale i j =
      ∑ k, dotGV V G i k * softmaxJacobian Q K scale i k j := by
  simp only [softmaxJacobian, mul_sub, Finset.sum_sub_distrib]
  have first : (∑ k, dotGV V G i k * (probability Q K scale i k *
      (if k = j then 1 else 0))) = dotGV V G i j * probability Q K scale i j := by
    simp
  rw [first]
  simp only [scoreGradient, correction, mul_sub, Finset.mul_sum]
  congr 1
  · ring
  · apply Finset.sum_congr rfl
    intro k _
    ring

/-- Forward and all three backward results, with every saved value constructed
from the inputs.  There is no premise asserting numerical kernel output. -/
theorem source_backward_eq_gradients (Q K V G : Tensor N D) (scale : ℝ) :
    savedOutput Q K V scale = attention Q K V scale ∧
    sourceDQ Q K V G (savedOutput Q K V scale) scale
      (savedNormalizer Q K scale) (savedMax Q K scale) = gradientQ Q K V G scale ∧
    sourceDK Q K V G (savedOutput Q K V scale) scale
      (savedNormalizer Q K scale) (savedMax Q K scale) = gradientK Q K V G scale ∧
    sourceDV Q K G scale (savedNormalizer Q K scale) (savedMax Q K scale) =
      gradientV Q K G scale := by
  refine ⟨savedOutput_eq_attention Q K V scale, ?_, ?_, ?_⟩
  · funext i e
    simp only [sourceDQ, gradientQ, sourceDS_eq_scoreGradient]
  · funext j e
    simp only [sourceDK, gradientK, sourceDS_eq_scoreGradient]
  · funext j e
    simp only [sourceDV, gradientV, preprocessDO]
    apply Finset.sum_congr rfl
    intro i _
    rw [← mul_div_assoc, mul_div_right_comm, sourceP_div_normalizer]

/-- Restrict ordinary memory-indexed arrays to the mathematical dimensions. -/
def restrictTensor (N D : Nat) (x : Nat → Nat → ℝ) : Tensor N D :=
  fun i e => x i.val e.val

#axiomsClean partition_pos
#axiomsClean probability_sum
#axiomsClean savedMax_ge
#axiomsClean savedMax_attained
#axiomsClean savedNormalizer_pos
#axiomsClean savedOutput_eq_attention
#axiomsClean preprocessDelta_eq
#axiomsClean sourceDS_eq_scoreGradient
#axiomsClean scoreGradient_eq_jacobian_pullback
#axiomsClean source_backward_eq_gradients
#specNonCircular attention avoiding [sourceP, sourceDS, sourceDQ, sourceDK, sourceDV]
#specNonCircular gradientQ avoiding [sourceP, sourceDS, sourceDQ, sourceDK, sourceDV]
#specNonCircular gradientK avoiding [sourceP, sourceDS, sourceDQ, sourceDK, sourceDV]
#specNonCircular gradientV avoiding [sourceP, sourceDS, sourceDQ, sourceDK, sourceDV]

end VeriTile.Bench.Repairs.TritonAttention.Whole001.Math
