import bench.repairs.tritonbench_g.parallel_attention.Proofs.Specification

namespace VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
open scoped BigOperators

def affineInputs (x dx : Inputs ℝ) (a : ℝ) : Inputs ℝ :=
  ⟨fun t e => x.q t e+a*dx.q t e,
   fun j e => x.k j e+a*dx.k j e,
   fun j p => x.v j p+a*dx.v j p⟩

@[simp] theorem affineInputs_zero (x dx : Inputs ℝ) : affineInputs x dx 0=x := by
  cases x
  simp [affineInputs]

theorem affine_derivative (x dx : ℝ) : HasDerivAt (fun a : ℝ => x+a*dx) dx 0 := by
  simpa using ((hasDerivAt_id (0:ℝ)).mul_const dx).const_add x

def scoreDirection (K : Nat) (scale : ℝ) (x dx : Inputs ℝ) (t j : Nat) : ℝ :=
  scale * ∑ e∈Finset.range K,(dx.q t e*x.k j e+x.q t e*dx.k j e)

theorem score_derivative (K : Nat) (scale : ℝ) (x dx : Inputs ℝ) (t j : Nat) :
    HasDerivAt (fun a : ℝ => score K scale (affineInputs x dx a) t j)
      (scoreDirection K scale x dx t j) 0 := by
  unfold score scoreDirection
  apply HasDerivAt.const_mul
  apply HasDerivAt.fun_sum
  intro e he
  convert (affine_derivative (x.q t e) (dx.q t e)).mul
    (affine_derivative (x.k j e) (dx.k j e)) using 1 <;> simp

def cotangentDirection (V : Nat) (dx : Inputs ℝ) (g : Cotangent ℝ) (t j : Nat) : ℝ :=
  ∑ p∈Finset.range V,g.output t p*dx.v j p

theorem cotangent_derivative (V : Nat) (x dx : Inputs ℝ) (g : Cotangent ℝ) (t j : Nat) :
    HasDerivAt (fun a : ℝ => scoreCotangent V (affineInputs x dx a) g t j)
      (cotangentDirection V dx g t j) 0 := by
  unfold scoreCotangent cotangentDirection
  apply HasDerivAt.add_const
  apply HasDerivAt.fun_sum
  intro p hp
  exact (affine_derivative (x.v j p) (dx.v j p)).const_mul (g.output t p)

def pairLoss (T K V : Nat) (scale : ℝ) (x : Inputs ℝ) (g : Cotangent ℝ) : ℝ :=
  ∑ t∈Finset.range T,∑ j∈Finset.range T,
    weight K scale x t j * scoreCotangent V x g t j

def lossSpec (T K V : Nat) (scale : ℝ) (x : Inputs ℝ) (g : Cotangent ℝ) : ℝ :=
  ∑ t∈Finset.range T,
    ((∑ p∈Finset.range V,g.output t p*outputSpec T K scale x t p)+
      g.normalizer t*normalizerSpec T K scale x t)

/-- The causal-pair objective is exactly the cotangent pairing with independent O/Z. -/
theorem loss_pair_form (T K V : Nat) (scale : ℝ) (x : Inputs ℝ) (g : Cotangent ℝ) :
    lossSpec T K V scale x g=pairLoss T K V scale x g := by
  unfold lossSpec pairLoss
  apply Finset.sum_congr rfl
  intro t ht
  unfold outputSpec normalizerSpec scoreCotangent
  simp_rw [Finset.mul_sum,mul_add,Finset.sum_add_distrib,Finset.mul_sum]
  congr 1
  · rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro j hj
    apply Finset.sum_congr rfl
    intro p hp
    ring
  · apply Finset.sum_congr rfl
    intro j hj
    ring

theorem pair_term_derivative (K V : Nat) (scale : ℝ) (x dx : Inputs ℝ)
    (g : Cotangent ℝ) (t j : Nat) :
    HasDerivAt (fun a : ℝ => weight K scale (affineInputs x dx a) t j *
      scoreCotangent V (affineInputs x dx a) g t j)
      ((∑ e∈Finset.range K,gradientCoefficient K V scale x g t j *
        (dx.q t e*x.k j e+x.q t e*dx.k j e))+
       (∑ p∈Finset.range V,weight K scale x t j*g.output t p*dx.v j p)) 0 := by
  by_cases causal : j≤t
  · have h := ((score_derivative K scale x dx t j).pow 2).mul
      (cotangent_derivative V x dx g t j)
    convert h using 1
    · ext a
      simp [weight,causal]
    · simp only [Pi.pow_apply,affineInputs_zero]
      simp only [gradientCoefficient,weight,if_pos causal,scoreDirection,cotangentDirection]
      simp only [Finset.mul_sum,Finset.sum_mul]
      congr 1
      · apply Finset.sum_congr rfl
        intro e he
        ring
      · apply Finset.sum_congr rfl
        intro p hp
        ring
  · simpa [weight,gradientCoefficient,causal] using (hasDerivAt_const (0:ℝ) (0:ℝ))

/-- Calculus establishes that the independent Q/K/V formulas are the VJP of
causal quadratic attention. It makes no differentiability claim about rounding. -/
theorem independent_vjp (T K V : Nat) (scale : ℝ) (x dx : Inputs ℝ) (g : Cotangent ℝ) :
    HasDerivAt (fun a : ℝ => lossSpec T K V scale (affineInputs x dx a) g)
      ((∑ t∈Finset.range T,∑ e∈Finset.range K,dx.q t e*querySpec T K V scale x g t e)+
       (∑ j∈Finset.range T,∑ e∈Finset.range K,dx.k j e*keySpec T K V scale x g j e)+
       (∑ j∈Finset.range T,∑ p∈Finset.range V,dx.v j p*valueSpec T K scale x g j p)) 0 := by
  simp_rw [loss_pair_form]
  rw [← vjp_pairing]
  unfold pairLoss directionalSpec
  apply HasDerivAt.fun_sum
  intro t ht
  apply HasDerivAt.fun_sum
  intro j hj
  exact pair_term_derivative K V scale x dx g t j

#axiomsClean score_derivative
#axiomsClean cotangent_derivative
#axiomsClean loss_pair_form
#axiomsClean pair_term_derivative
#axiomsClean independent_vjp
end
end VeriTile.Bench.Repairs.ParallelAttention
