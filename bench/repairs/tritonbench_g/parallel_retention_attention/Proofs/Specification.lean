import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.Real

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention
noncomputable section
open scoped BigOperators

/-- Independent causal coefficient. There is no softmax or normalization. -/
def weight (h t n : Nat) : ℝ := if n≤t then gamma h^(t-n) else 0

def outputSpec (T DK DV h : Nat) (q k : Fin T → Fin DK → ℝ)
    (v : Fin T → Fin DV → ℝ) (t : Fin T) (f : Fin DV) : ℝ :=
  (Real.sqrt (DK:ℝ))⁻¹ * ∑ n : Fin T,weight h t n*(∑ e : Fin DK,q t e*k n e)*v n f

def querySpec (T DK DV h : Nat) (k : Fin T → Fin DK → ℝ)
    (v g : Fin T → Fin DV → ℝ) (t : Fin T) (e : Fin DK) : ℝ :=
  (Real.sqrt (DK:ℝ))⁻¹ * ∑ n : Fin T,weight h t n*k n e*(∑ f : Fin DV,g t f*v n f)

def keySpec (T DK DV h : Nat) (q : Fin T → Fin DK → ℝ)
    (v g : Fin T → Fin DV → ℝ) (n : Fin T) (e : Fin DK) : ℝ :=
  (Real.sqrt (DK:ℝ))⁻¹ * ∑ t : Fin T,weight h t n*q t e*(∑ f : Fin DV,g t f*v n f)

def valueSpec (T DK DV h : Nat) (q k : Fin T → Fin DK → ℝ)
    (g : Fin T → Fin DV → ℝ) (n : Fin T) (f : Fin DV) : ℝ :=
  (Real.sqrt (DK:ℝ))⁻¹ * ∑ t : Fin T,weight h t n*g t f*(∑ e : Fin DK,q t e*k n e)

/-- The exact first-order term of the mathematical forward polynomial.
Inputs, directions and cotangent are arbitrary reals, including empty types. -/
def directionalSpec (T DK DV h : Nat) (q k dq dk : Fin T → Fin DK → ℝ)
    (v g dv : Fin T → Fin DV → ℝ) : ℝ :=
  ∑ t : Fin T,∑ n : Fin T,∑ e : Fin DK,∑ f : Fin DV,(Real.sqrt (DK:ℝ))⁻¹*weight h t n*g t f*
    (dq t e*k n e*v n f+q t e*dk n e*v n f+q t e*k n e*dv n f)

/-- The three independently stated gradients are the VJP of the causal
trilinear polynomial, not gradients of a source recurrence. -/
theorem vjp_pairing (T DK DV h : Nat) (q k dq dk : Fin T → Fin DK → ℝ)
    (v g dv : Fin T → Fin DV → ℝ) :
    directionalSpec T DK DV h q k dq dk v g dv =
      (∑ t : Fin T,∑ e : Fin DK,dq t e*querySpec T DK DV h k v g t e)+
      (∑ n : Fin T,∑ e : Fin DK,dk n e*keySpec T DK DV h q v g n e)+
      (∑ n : Fin T,∑ f : Fin DV,dv n f*valueSpec T DK DV h q k g n f) := by
  unfold directionalSpec querySpec keySpec valueSpec
  simp_rw [mul_add,Finset.sum_add_distrib,Finset.mul_sum]
  congr 1
  · congr 1
    · rw [Finset.sum_congr rfl (fun t _ => Finset.sum_comm)]
      apply Finset.sum_congr rfl
      intro t ht
      apply Finset.sum_congr rfl
      intro e he
      apply Finset.sum_congr rfl
      intro n hn
      apply Finset.sum_congr rfl
      intro f hf
      ring
    · rw [Finset.sum_comm]
      apply Finset.sum_congr rfl
      intro n hn
      rw [Finset.sum_comm]
      apply Finset.sum_congr rfl
      intro e he
      apply Finset.sum_congr rfl
      intro t ht
      apply Finset.sum_congr rfl
      intro f hf
      ring
  · rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro n hn
    rw [Finset.sum_congr rfl (fun t _ => Finset.sum_comm),Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro f hf
    apply Finset.sum_congr rfl
    intro t ht
    apply Finset.sum_congr rfl
    intro e he
    ring

#axiomsClean vjp_pairing
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention
