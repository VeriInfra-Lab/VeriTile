import bench.repairs.tritonbench_g.parallel_attention.Proofs.Types

namespace VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
open scoped BigOperators

/-- Logical input elements, independent of physical memory, tiles and execution. -/
def score (K : Nat) (scale : ℝ) (x : Inputs ℝ) (t j : Nat) : ℝ :=
  scale * ∑ e ∈ Finset.range K, x.q t e * x.k j e

def weight (K : Nat) (scale : ℝ) (x : Inputs ℝ) (t j : Nat) : ℝ :=
  if j ≤ t then (score K scale x t j)^2 else 0

@[standalone_spec] def outputSpec (T K : Nat) (scale : ℝ) (x : Inputs ℝ) (t p : Nat) : ℝ :=
  ∑ j ∈ Finset.range T, weight K scale x t j * x.v j p

@[standalone_spec] def normalizerSpec (T K : Nat) (scale : ℝ) (x : Inputs ℝ) (t : Nat) : ℝ :=
  ∑ j ∈ Finset.range T, weight K scale x t j

/-- Both public scale branches; no native rounding identity is asserted. -/
def selectedScale (K : Nat) (useScale : Bool) : ℝ :=
  if useScale then (Real.sqrt (K : ℝ))⁻¹ else 1

/-- Return-both has priority over the normalization flag. -/
@[standalone_spec] def publicSpec (T K : Nat) (scale eps : ℝ) (x : Inputs ℝ)
    (returnBoth normalize : Bool) : PublicResult ℝ :=
  let o := outputSpec T K scale x
  let z := normalizerSpec T K scale x
  if returnBoth then .both o z
  else if normalize then .single (fun t p => o t p / (z t + eps))
  else .single o

theorem return_both_priority (T K : Nat) (scale eps : ℝ) (x : Inputs ℝ) (normalize : Bool) :
    publicSpec T K scale eps x true normalize =
      .both (outputSpec T K scale x) (normalizerSpec T K scale x) := by
  simp [publicSpec]

/-- Cotangent of the score before the quadratic map, including the normalizer. -/
def scoreCotangent (V : Nat) (x : Inputs ℝ) (g : Cotangent ℝ) (t j : Nat) : ℝ :=
  (∑ p ∈ Finset.range V, g.output t p * x.v j p) + g.normalizer t

def gradientCoefficient (K V : Nat) (scale : ℝ) (x : Inputs ℝ)
    (g : Cotangent ℝ) (t j : Nat) : ℝ :=
  if j ≤ t then 2 * scale * scoreCotangent V x g t j * score K scale x t j else 0

@[standalone_spec] def querySpec (T K V : Nat) (scale : ℝ) (x : Inputs ℝ) (g : Cotangent ℝ) (t e : Nat) : ℝ :=
  ∑ j ∈ Finset.range T, gradientCoefficient K V scale x g t j * x.k j e

@[standalone_spec] def keySpec (T K V : Nat) (scale : ℝ) (x : Inputs ℝ) (g : Cotangent ℝ) (j e : Nat) : ℝ :=
  ∑ t ∈ Finset.range T, gradientCoefficient K V scale x g t j * x.q t e

@[standalone_spec] def valueSpec (T K : Nat) (scale : ℝ) (x : Inputs ℝ) (g : Cotangent ℝ) (j p : Nat) : ℝ :=
  ∑ t ∈ Finset.range T, weight K scale x t j * g.output t p

/-- First-order variation, independently expanded by causal pair and feature. -/
@[standalone_spec] def directionalSpec (T K V : Nat) (scale : ℝ) (x dx : Inputs ℝ) (g : Cotangent ℝ) : ℝ :=
  ∑ t ∈ Finset.range T, ∑ j ∈ Finset.range T,
    ((∑ e ∈ Finset.range K,
      gradientCoefficient K V scale x g t j * (dx.q t e * x.k j e + x.q t e * dx.k j e)) +
    (∑ p ∈ Finset.range V, weight K scale x t j * g.output t p * dx.v j p))

/-- The three independent gradient tensors pair with every input direction. -/
theorem vjp_pairing (T K V : Nat) (scale : ℝ) (x dx : Inputs ℝ) (g : Cotangent ℝ) :
    directionalSpec T K V scale x dx g =
      (∑ t ∈ Finset.range T, ∑ e ∈ Finset.range K, dx.q t e * querySpec T K V scale x g t e) +
      (∑ j ∈ Finset.range T, ∑ e ∈ Finset.range K, dx.k j e * keySpec T K V scale x g j e) +
      (∑ j ∈ Finset.range T, ∑ p ∈ Finset.range V, dx.v j p * valueSpec T K scale x g j p) := by
  unfold directionalSpec querySpec keySpec valueSpec
  simp_rw [mul_add, Finset.sum_add_distrib, Finset.mul_sum]
  congr 1
  · congr 1
    · apply Finset.sum_congr rfl
      intro t ht
      rw [Finset.sum_comm]
      apply Finset.sum_congr rfl
      intro e he
      apply Finset.sum_congr rfl
      intro j hj
      ring
    · rw [Finset.sum_comm]
      apply Finset.sum_congr rfl
      intro j hj
      rw [Finset.sum_comm]
      apply Finset.sum_congr rfl
      intro e he
      apply Finset.sum_congr rfl
      intro t ht
      ring
  · rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro j hj
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro p hp
    apply Finset.sum_congr rfl
    intro t ht
    ring

/-- Unnormalized single-output calls supply zero normalizer cotangent. -/
def outputOnlyCotangent (g : Nat → Nat → ℝ) : Cotangent ℝ := ⟨g, fun _ => 0⟩

/-- The mathematical normalization VJP; the differentiability theorem requires
nonzero denominators, while source FP division remains a separate scalar port. -/
def normalizedCotangent (T K V : Nat) (scale eps : ℝ) (x : Inputs ℝ)
    (g : Nat → Nat → ℝ) : Cotangent ℝ :=
  ⟨fun t p => g t p / (normalizerSpec T K scale x t + eps),
   fun t => -(∑ p ∈ Finset.range V, g t p * outputSpec T K scale x t p) /
     (normalizerSpec T K scale x t + eps)^2⟩


end
end VeriTile.Bench.Repairs.ParallelAttention
