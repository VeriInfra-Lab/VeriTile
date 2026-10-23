import bench.repairs.tritonbench_g.triton_attention.Whole001.Proofs.Math
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul

namespace VeriTile.Bench.Repairs.TritonAttention.Whole001.Math
open scoped BigOperators
noncomputable section
variable {N D : Nat}

def perturb (X dX : Tensor N D) (t : ℝ) : Tensor N D :=
  fun i e => X i e + t * dX i e

def scoreTangent (Q K dQ dK : Tensor N D) (scale : ℝ) (i j : Fin N) : ℝ :=
  (∑ e, (dQ i e * K j e + Q i e * dK j e)) * scale

def probabilityTangent (Q K dQ dK : Tensor N D) (scale : ℝ) (i j : Fin N) : ℝ :=
  probability Q K scale i j * (scoreTangent Q K dQ dK scale i j -
    ∑ k, probability Q K scale i k * scoreTangent Q K dQ dK scale i k)

@[simp] theorem perturb_zero (X dX : Tensor N D) : perturb X dX 0 = X := by
  funext i e
  simp [perturb]

theorem perturb_hasDerivAt (X dX : Tensor N D) (i : Fin N) (e : Fin D) :
    HasDerivAt (fun t => perturb X dX t i e) (dX i e) 0 := by
  simpa [perturb] using ((hasDerivAt_id (0 : ℝ)).mul_const (dX i e)).const_add (X i e)

theorem score_hasDerivAt (Q K dQ dK : Tensor N D) (scale : ℝ) (i j : Fin N) :
    HasDerivAt (fun t => score (perturb Q dQ t) (perturb K dK t) scale i j)
      (scoreTangent Q K dQ dK scale i j) 0 := by
  have h := (HasDerivAt.fun_sum (u := Finset.univ) fun e _ =>
    (perturb_hasDerivAt Q dQ i e).mul (perturb_hasDerivAt K dK j e)).mul_const scale
  simpa [score,scoreTangent] using h

theorem causalWeight_hasDerivAt (Q K dQ dK : Tensor N D) (scale : ℝ) (i j : Fin N) :
    HasDerivAt (fun t => causalWeight (perturb Q dQ t) (perturb K dK t) scale i j)
      (causalWeight Q K scale i j * scoreTangent Q K dQ dK scale i j) 0 := by
  by_cases h : j ≤ i
  · simpa [causalWeight,h] using (score_hasDerivAt Q K dQ dK scale i j).exp
  · simpa [causalWeight,h] using (hasDerivAt_const (0 : ℝ) (0 : ℝ))

theorem probability_hasDerivAt (Q K dQ dK : Tensor N D) (scale : ℝ) (i j : Fin N) :
    HasDerivAt (fun t => probability (perturb Q dQ t) (perturb K dK t) scale i j)
      (probabilityTangent Q K dQ dK scale i j) 0 := by
  have hp := HasDerivAt.fun_sum (u := Finset.univ) fun k _ =>
    causalWeight_hasDerivAt Q K dQ dK scale i k
  have h := (causalWeight_hasDerivAt Q K dQ dK scale i j).div hp
    (by simpa [partition] using ne_of_gt (partition_pos Q K scale i))
  simp only [perturb_zero] at h
  convert h using 1
  dsimp [probabilityTangent,probability]
  have hz := ne_of_gt (partition_pos Q K scale i)
  simp_rw [div_mul_eq_mul_div, ← Finset.sum_div]
  change _ = (_ * partition Q K scale i - _ * _) / (partition Q K scale i)^2
  field_simp
  <;> ring

def attentionTangent (Q K V dQ dK dV : Tensor N D) (scale : ℝ) (i : Fin N)
    (e : Fin D) : ℝ :=
  ∑ j, (probabilityTangent Q K dQ dK scale i j * V j e +
    probability Q K scale i j * dV j e)

theorem attention_hasDerivAt (Q K V dQ dK dV : Tensor N D) (scale : ℝ)
    (i : Fin N) (e : Fin D) :
    HasDerivAt (fun t => attention (perturb Q dQ t) (perturb K dK t)
      (perturb V dV t) scale i e) (attentionTangent Q K V dQ dK dV scale i e) 0 := by
  have h := HasDerivAt.fun_sum (u := Finset.univ) fun j _ =>
    (probability_hasDerivAt Q K dQ dK scale i j).mul (perturb_hasDerivAt V dV j e)
  simpa [attention,attentionTangent] using h

def loss (Q K V G : Tensor N D) (scale : ℝ) : ℝ :=
  ∑ i, ∑ e, G i e * attention Q K V scale i e

theorem loss_hasDerivAt_tangent (Q K V G dQ dK dV : Tensor N D) (scale : ℝ) :
    HasDerivAt (fun t => loss (perturb Q dQ t) (perturb K dK t)
      (perturb V dV t) G scale)
      (∑ i, ∑ e, G i e * attentionTangent Q K V dQ dK dV scale i e) 0 := by
  exact HasDerivAt.fun_sum (u := Finset.univ) fun i _ => HasDerivAt.fun_sum (u := Finset.univ) fun e _ =>
    (attention_hasDerivAt Q K V dQ dK dV scale i e).const_mul (G i e)

/-- Algebraic adjoint of the derivative just proved for the causal softmax. -/
theorem centered_pairing (p x t : Fin N → ℝ) :
    (∑ j, x j * (p j * (t j - ∑ k, p k * t k))) =
      ∑ j, (p j * (x j - ∑ k, p k * x k)) * t j := by
  simp only [mul_sub, sub_mul, Finset.sum_sub_distrib]
  have h1 : (∑ j, x j * (p j * t j)) = ∑ j, p j * x j * t j := by
    apply Finset.sum_congr rfl
    intro j _
    ring
  have h2 : (∑ j, x j * (p j * ∑ k, p k * t k)) =
      (∑ k, p k * x k) * ∑ k, p k * t k := by
    rw [Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro j _
    ring
  have h3 : (∑ j, p j * (∑ k, p k * x k) * t j) =
      (∑ k, p k * x k) * ∑ k, p k * t k := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro j _
    ring
  rw [h1,h2,h3]

theorem tangent_pairing (Q K V G dQ dK dV : Tensor N D) (scale : ℝ) :
    (∑ i, ∑ e, G i e * attentionTangent Q K V dQ dK dV scale i e) =
      (∑ i, ∑ j, scoreGradient Q K V G scale i j * scoreTangent Q K dQ dK scale i j) +
      ∑ j, ∑ e, gradientV Q K G scale j e * dV j e := by
  have row (i : Fin N) :
      (∑ e, G i e * attentionTangent Q K V dQ dK dV scale i e) =
      (∑ j, scoreGradient Q K V G scale i j * scoreTangent Q K dQ dK scale i j) +
      ∑ j, ∑ e, (probability Q K scale i j * G i e) * dV j e := by
    simp only [attentionTangent,Finset.mul_sum,mul_add,Finset.sum_add_distrib]
    rw [Finset.sum_comm,Finset.sum_comm (f := fun e j => G i e * (probability Q K scale i j * dV j e))]
    congr 1
    · have rearrange : (∑ j, ∑ e, G i e * (probabilityTangent Q K dQ dK scale i j * V j e)) =
          ∑ j, dotGV V G i j * probabilityTangent Q K dQ dK scale i j := by
        apply Finset.sum_congr rfl
        intro j _
        simp only [dotGV,Finset.sum_mul]
        apply Finset.sum_congr rfl
        intro e _
        ring
      rw [rearrange]
      exact centered_pairing (probability Q K scale i) (dotGV V G i)
        (scoreTangent Q K dQ dK scale i)
    · apply Finset.sum_congr rfl
      intro j _
      apply Finset.sum_congr rfl
      intro e _
      ring
  simp_rw [row]
  rw [Finset.sum_add_distrib]
  congr 1
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro j _
  rw [Finset.sum_comm]
  simp only [gradientV,Finset.sum_mul]

theorem score_tangent_pairing (Q K V G dQ dK : Tensor N D) (scale : ℝ) :
    (∑ i, ∑ j, scoreGradient Q K V G scale i j * scoreTangent Q K dQ dK scale i j) =
      (∑ i, ∑ e, gradientQ Q K V G scale i e * dQ i e) +
      ∑ j, ∑ e, gradientK Q K V G scale j e * dK j e := by
  have term (i j : Fin N) :
      scoreGradient Q K V G scale i j * scoreTangent Q K dQ dK scale i j =
      (∑ e, ((scoreGradient Q K V G scale i j * scale) * K j e) * dQ i e) +
      ∑ e, ((scoreGradient Q K V G scale i j * scale) * Q i e) * dK j e := by
    simp only [scoreTangent,Finset.sum_mul,Finset.mul_sum,← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro e _
    ring
  simp_rw [term,Finset.sum_add_distrib]
  congr 1
  · apply Finset.sum_congr rfl
    intro i _
    rw [Finset.sum_comm]
    simp only [gradientQ,Finset.sum_mul]
  · rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro j _
    rw [Finset.sum_comm]
    simp only [gradientK,Finset.sum_mul]

/-- The three independent target tensors are the VJP of causal attention:
for every simultaneous Q/K/V direction, the loss derivative is their pairing.
Scale is held fixed, matching the returned None scale cotangent. -/
theorem attention_vjp (Q K V G dQ dK dV : Tensor N D) (scale : ℝ) :
    HasDerivAt (fun t => loss (perturb Q dQ t) (perturb K dK t)
      (perturb V dV t) G scale)
      ((∑ i, ∑ e, gradientQ Q K V G scale i e * dQ i e) +
       (∑ j, ∑ e, gradientK Q K V G scale j e * dK j e) +
        ∑ j, ∑ e, gradientV Q K G scale j e * dV j e) 0 := by
  rw [← score_tangent_pairing, ← tangent_pairing]
  exact loss_hasDerivAt_tangent Q K V G dQ dK dV scale

#axiomsClean attention_vjp
#axiomsClean probability_hasDerivAt
#axiomsClean attention_hasDerivAt
#axiomsClean loss_hasDerivAt_tangent
end
end VeriTile.Bench.Repairs.TritonAttention.Whole001.Math
