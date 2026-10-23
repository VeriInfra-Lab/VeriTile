import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.ScalarOutput
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.MathTarget
noncomputable section
open scoped BigOperators

structure Variation where
  q : Nat → Nat → Nat → ℝ
  k : Nat → Nat → Nat → ℝ
  v : Nat → Nat → Nat → ℝ
  seed : Nat → Nat → Nat → ℝ

/-- Direct polynomial differential, summed over all independent state coordinates.
Each coordinate includes the whole time series and its final-state loss. -/
def fullVariation (c : Shape) (xs : Inputs ℝ) (d : Variation) (scale : ℝ)
    (hasSeed : Bool) : ℝ :=
  ∑ h ∈ Finset.range c.heads, ∑ e ∈ Finset.range c.K, ∑ j ∈ Finset.range c.V,
    lossVariation (headRate c h) (if hasSeed then xs.seed h e j else 0)
      (if hasSeed then d.seed h e j else 0) (xs.finalIncoming h e j) scale
      (fun t => xs.q h t e) (fun t => xs.k h t e) (fun t => xs.v h t j)
      (fun t => xs.incoming h t j) (fun t => d.q h t e) (fun t => d.k h t e)
      (fun t => d.v h t j) c.T

def gradientPairing (c : Shape) (xs : Inputs ℝ) (d : Variation) (scale : ℝ)
    (hasSeed : Bool) : ℝ :=
  (∑ h ∈ Finset.range c.heads, ∑ t ∈ Finset.range c.T, ∑ e ∈ Finset.range c.K,
    queryGradient c xs scale hasSeed h t e*d.q h t e)+
  (∑ h ∈ Finset.range c.heads, ∑ t ∈ Finset.range c.T, ∑ e ∈ Finset.range c.K,
    keyGradient c xs scale h t e*d.k h t e)+
  (∑ h ∈ Finset.range c.heads, ∑ t ∈ Finset.range c.T, ∑ j ∈ Finset.range c.V,
    valueGradient c xs scale h t j*d.v h t j)+
  ∑ h ∈ Finset.range c.heads, ∑ e ∈ Finset.range c.K, ∑ j ∈ Finset.range c.V,
    initialGradient c xs scale h e j*(if hasSeed then d.seed h e j else 0)

theorem coordinate_gradient (c : Shape) (xs : Inputs ℝ) (d : Variation) (scale : ℝ)
    (hasSeed : Bool) (h e j : Nat) :
    lossVariation (headRate c h) (if hasSeed then xs.seed h e j else 0)
      (if hasSeed then d.seed h e j else 0) (xs.finalIncoming h e j) scale
      (fun t => xs.q h t e) (fun t => xs.k h t e) (fun t => xs.v h t j)
      (fun t => xs.incoming h t j) (fun t => d.q h t e) (fun t => d.k h t e)
      (fun t => d.v h t j) c.T=
    (∑ t ∈ Finset.range c.T,
      (scale*xs.incoming h t j*state c xs hasSeed h (t+1) e j)*d.q h t e)+
    (∑ t ∈ Finset.range c.T,
      (xs.v h t j*tokenCoefficient c xs scale h t e j)*d.k h t e)+
    (∑ t ∈ Finset.range c.T,
      (xs.k h t e*tokenCoefficient c xs scale h t e j)*d.v h t j)+
    initialGradient c xs scale h e j*(if hasSeed then d.seed h e j else 0) := by
  rw [coordinate_vjp]
  have hc (t : Nat) (ht : t<c.T) :
      denseCoefficient (headRate c h) (xs.finalIncoming h e j)
        (fun u => scale*xs.q h u e*xs.incoming h u j) t c.T=
      tokenCoefficient c xs scale h t e j := by
    exact (coefficient_dense _ _ _ t c.T ht).symm
  congr 2
  · congr 1
    apply Finset.sum_congr rfl
    intro t ht
    rw [hc t (Finset.mem_range.mp ht)]
    ring
  · apply Finset.sum_congr rfl
    intro t ht
    rw [hc t (Finset.mem_range.mp ht)]
    ring

/-- Bring time outside key/value coordinates without changing multiplicities. -/
theorem sum_time_first (K V T : Nat) (f : Nat → Nat → Nat → ℝ) :
    (∑ e ∈ Finset.range K, ∑ j ∈ Finset.range V, ∑ t ∈ Finset.range T, f t e j)=
      ∑ t ∈ Finset.range T, ∑ e ∈ Finset.range K, ∑ j ∈ Finset.range V, f t e j := by
  calc
    _ = ∑ e ∈ Finset.range K, ∑ t ∈ Finset.range T, ∑ j ∈ Finset.range V, f t e j := by
      apply Finset.sum_congr rfl
      intro e he
      rw [Finset.sum_comm]
    _ = _ := Finset.sum_comm

theorem whole_vjp (c : Shape) (xs : Inputs ℝ) (d : Variation) (scale : ℝ) (hasSeed : Bool) :
    fullVariation c xs d scale hasSeed=gradientPairing c xs d scale hasSeed := by
  unfold fullVariation
  simp_rw [coordinate_gradient,Finset.sum_add_distrib]
  unfold gradientPairing
  congr 2
  · congr 1
    · apply Finset.sum_congr rfl
      intro h hh
      rw [sum_time_first]
      apply Finset.sum_congr rfl
      intro t ht
      apply Finset.sum_congr rfl
      intro e he
      unfold queryGradient
      rw [mul_assoc,Finset.sum_mul,Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro j hj
      ring
    · apply Finset.sum_congr rfl
      intro h hh
      rw [sum_time_first]
      apply Finset.sum_congr rfl
      intro t ht
      apply Finset.sum_congr rfl
      intro e he
      simp only [keyGradient,Finset.sum_mul]
  · apply Finset.sum_congr rfl
    intro h hh
    rw [sum_time_first]
    apply Finset.sum_congr rfl
    intro t ht
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro j hj
    simp only [valueGradient,Finset.sum_mul]

#axiomsClean whole_vjp
#specNonCircular fullVariation avoiding [gradientPairing,whole_vjp]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.MathTarget

