import bench.repairs.tritonbench_g.parallel_attention.Proofs.Differential

namespace VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
open scoped BigOperators

def weightDirection (K : Nat) (scale : ℝ) (x dx : Inputs ℝ) (t j : Nat) : ℝ :=
  if j ≤ t then 2*score K scale x t j*scoreDirection K scale x dx t j else 0

def outputDirection (T K : Nat) (scale : ℝ) (x dx : Inputs ℝ) (t p : Nat) : ℝ :=
  ∑ j∈Finset.range T,(weightDirection K scale x dx t j*x.v j p+weight K scale x t j*dx.v j p)

def normalizerDirection (T K : Nat) (scale : ℝ) (x dx : Inputs ℝ) (t : Nat) : ℝ :=
  ∑ j∈Finset.range T,weightDirection K scale x dx t j

theorem weight_derivative (K : Nat) (scale : ℝ) (x dx : Inputs ℝ) (t j : Nat) :
    HasDerivAt (fun a : ℝ => weight K scale (affineInputs x dx a) t j)
      (weightDirection K scale x dx t j) 0 := by
  by_cases h : j ≤ t
  · simpa [weight,weightDirection,h,Pi.pow_apply] using (score_derivative K scale x dx t j).pow 2
  · simpa [weight,weightDirection,h] using hasDerivAt_const (0:ℝ) (0:ℝ)

theorem output_derivative (T K : Nat) (scale : ℝ) (x dx : Inputs ℝ) (t p : Nat) :
    HasDerivAt (fun a : ℝ => outputSpec T K scale (affineInputs x dx a) t p)
      (outputDirection T K scale x dx t p) 0 := by
  unfold outputSpec outputDirection
  apply HasDerivAt.fun_sum
  intro j hj
  convert (weight_derivative K scale x dx t j).mul (affine_derivative (x.v j p) (dx.v j p)) using 1 <;> simp [affineInputs]

theorem normalizer_derivative (T K : Nat) (scale : ℝ) (x dx : Inputs ℝ) (t : Nat) :
    HasDerivAt (fun a : ℝ => normalizerSpec T K scale (affineInputs x dx a) t)
      (normalizerDirection T K scale x dx t) 0 := by
  unfold normalizerSpec normalizerDirection
  apply HasDerivAt.fun_sum
  intro j hj
  exact weight_derivative K scale x dx t j

def lossDirection (T K V : Nat) (scale : ℝ) (x dx : Inputs ℝ) (g : Cotangent ℝ) : ℝ :=
  ∑ t∈Finset.range T,
    ((∑ p∈Finset.range V,g.output t p*outputDirection T K scale x dx t p)+
      g.normalizer t*normalizerDirection T K scale x dx t)

theorem loss_derivative_direct (T K V : Nat) (scale : ℝ) (x dx : Inputs ℝ) (g : Cotangent ℝ) :
    HasDerivAt (fun a : ℝ => lossSpec T K V scale (affineInputs x dx a) g)
      (lossDirection T K V scale x dx g) 0 := by
  unfold lossSpec lossDirection
  apply HasDerivAt.fun_sum
  intro t ht
  apply HasDerivAt.add
  · apply HasDerivAt.fun_sum
    intro p hp
    exact (output_derivative T K scale x dx t p).const_mul (g.output t p)
  · exact (normalizer_derivative T K scale x dx t).const_mul (g.normalizer t)

def normalizedLoss (T K V : Nat) (scale eps : ℝ) (x : Inputs ℝ) (g : Nat → Nat → ℝ) : ℝ :=
  ∑ t∈Finset.range T,∑ p∈Finset.range V,
    g t p*(outputSpec T K scale x t p/(normalizerSpec T K scale x t+eps))

theorem normalization_chain (T K V : Nat) (scale eps : ℝ) (x dx : Inputs ℝ)
    (g : Nat → Nat → ℝ) (nonzero : ∀ t,t < T → normalizerSpec T K scale x t+eps ≠ 0) :
    HasDerivAt (fun a : ℝ => normalizedLoss T K V scale eps (affineInputs x dx a) g)
      (lossDirection T K V scale x dx (normalizedCotangent T K V scale eps x g)) 0 := by
  have h : HasDerivAt (fun a : ℝ => normalizedLoss T K V scale eps (affineInputs x dx a) g)
      (∑ t∈Finset.range T,∑ p∈Finset.range V,g t p*
        ((outputDirection T K scale x dx t p*(normalizerSpec T K scale x t+eps)-
          outputSpec T K scale x t p*normalizerDirection T K scale x dx t)/
          (normalizerSpec T K scale x t+eps)^2)) 0 := by
    unfold normalizedLoss
    apply HasDerivAt.fun_sum
    intro t ht
    apply HasDerivAt.fun_sum
    intro p hp
    convert ((output_derivative T K scale x dx t p).div
      ((normalizer_derivative T K scale x dx t).add_const eps)
      (by simpa using nonzero t (Finset.mem_range.mp ht))).const_mul (g t p) using 1 <;> simp
  convert h using 1
  unfold lossDirection normalizedCotangent
  apply Finset.sum_congr rfl
  intro t ht
  simp only
  simp only [neg_div,Finset.sum_div,neg_mul,Finset.sum_mul,← Finset.sum_neg_distrib]
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro p hp
  have hn := nonzero t (Finset.mem_range.mp ht)
  field_simp [hn]
  <;> ring

/-- The normalization wrapper has the stated Q/K/V VJP whenever each Real
normalization denominator is nonzero. Return-both and raw-output use their own
cotangents and do not acquire this extra mathematical domain condition. -/
theorem normalized_vjp (T K V : Nat) (scale eps : ℝ) (x dx : Inputs ℝ)
    (g : Nat → Nat → ℝ) (nonzero : ∀ t,t < T → normalizerSpec T K scale x t+eps ≠ 0) :
    HasDerivAt (fun a : ℝ => normalizedLoss T K V scale eps (affineInputs x dx a) g)
      ((∑ t∈Finset.range T,∑ e∈Finset.range K,dx.q t e*querySpec T K V scale x
          (normalizedCotangent T K V scale eps x g) t e)+
       (∑ j∈Finset.range T,∑ e∈Finset.range K,dx.k j e*keySpec T K V scale x
          (normalizedCotangent T K V scale eps x g) j e)+
       (∑ j∈Finset.range T,∑ p∈Finset.range V,dx.v j p*valueSpec T K scale x
          (normalizedCotangent T K V scale eps x g) j p)) 0 := by
  have equal := (loss_derivative_direct T K V scale x dx (normalizedCotangent T K V scale eps x g)).unique
    (independent_vjp T K V scale x dx (normalizedCotangent T K V scale eps x g))
  rw [← equal]
  exact normalization_chain T K V scale eps x dx g nonzero

#axiomsClean output_derivative
#axiomsClean normalizer_derivative
#axiomsClean normalization_chain
#axiomsClean normalized_vjp
end
end VeriTile.Bench.Repairs.ParallelAttention
