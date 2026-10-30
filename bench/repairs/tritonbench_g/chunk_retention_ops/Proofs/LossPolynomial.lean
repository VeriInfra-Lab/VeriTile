import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.WholeVJP
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.MathTarget
noncomputable section
open scoped BigOperators

/-- Perturb primal inputs only; both cotangents stay fixed. -/
def perturb (xs : Inputs ℝ) (d : Variation) (z : ℝ) : Inputs ℝ :=
  {xs with
    q := fun h t e => xs.q h t e+z*d.q h t e
    k := fun h t e => xs.k h t e+z*d.k h t e
    v := fun h t j => xs.v h t j+z*d.v h t j
    seed := fun h e j => xs.seed h e j+z*d.seed h e j}

/-- Exact finite quadratic expansion of the independent state formula. -/
theorem weighted_polynomial (rate seed dSeed z : ℝ)
    (k v dk dv : Nat→ℝ) (n : Nat) :
    weighted rate (seed+z*dSeed) (fun t => (k t+z*dk t)*(v t+z*dv t)) n=
      weighted rate seed (fun t=>k t*v t) n+
      z*stateVariation rate dSeed k v dk dv n+
      z^2*weighted rate 0 (fun t=>dk t*dv t) n := by
  have hs :
      (∑ t∈Finset.range n,rate^(n-1-t)*((k t+z*dk t)*(v t+z*dv t)))=
      (∑ t∈Finset.range n,rate^(n-1-t)*(k t*v t))+
      z*(∑ t∈Finset.range n,rate^(n-1-t)*(dk t*v t+k t*dv t))+
      z^2*(∑ t∈Finset.range n,rate^(n-1-t)*(dk t*dv t)) := by
    simp only [Finset.mul_sum,← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro t ht
    ring
  unfold stateVariation weighted
  rw [hs]
  ring

def coordinateLoss (rate seed terminal scale : ℝ)
    (q k v incoming : Nat→ℝ) (n : Nat) : ℝ :=
  (∑ t∈Finset.range n,incoming t*scale*
    (q t*weighted rate seed (fun u=>k u*v u) (t+1)))+
    terminal*weighted rate seed (fun u=>k u*v u) n

def coordinateSecond (rate dSeed terminal scale : ℝ)
    (q k v incoming dq dk dv : Nat→ℝ) (n : Nat) : ℝ :=
  (∑ t∈Finset.range n,incoming t*scale*
    (dq t*stateVariation rate dSeed k v dk dv (t+1)+
     q t*weighted rate 0 (fun u=>dk u*dv u) (t+1)))+
    terminal*weighted rate 0 (fun u=>dk u*dv u) n

def coordinateThird (rate scale : ℝ) (incoming dq dk dv : Nat→ℝ) (n : Nat) : ℝ :=
  ∑ t∈Finset.range n,incoming t*scale*
    (dq t*weighted rate 0 (fun u=>dk u*dv u) (t+1))

/-- The existing lossVariation is the first coefficient of an exact cubic. -/
theorem coordinate_polynomial (rate seed dSeed terminal scale z : ℝ)
    (q k v incoming dq dk dv : Nat→ℝ) (n : Nat) :
    coordinateLoss rate (seed+z*dSeed) terminal scale
      (fun t=>q t+z*dq t) (fun t=>k t+z*dk t) (fun t=>v t+z*dv t) incoming n=
    coordinateLoss rate seed terminal scale q k v incoming n+
      z*lossVariation rate seed dSeed terminal scale q k v incoming dq dk dv n+
      z^2*coordinateSecond rate dSeed terminal scale q k v incoming dq dk dv n+
      z^3*coordinateThird rate scale incoming dq dk dv n := by
  have hs :
      (∑ t∈Finset.range n,incoming t*scale*((q t+z*dq t)*
        weighted rate (seed+z*dSeed) (fun u=>(k u+z*dk u)*(v u+z*dv u)) (t+1)))=
      (∑ t∈Finset.range n,incoming t*scale*(q t*weighted rate seed (fun u=>k u*v u) (t+1)))+
      z*(∑ t∈Finset.range n,incoming t*scale*(dq t*weighted rate seed (fun u=>k u*v u) (t+1)+
        q t*stateVariation rate dSeed k v dk dv (t+1)))+
      z^2*(∑ t∈Finset.range n,incoming t*scale*(dq t*stateVariation rate dSeed k v dk dv (t+1)+
        q t*weighted rate 0 (fun u=>dk u*dv u) (t+1)))+
      z^3*(∑ t∈Finset.range n,incoming t*scale*(dq t*weighted rate 0 (fun u=>dk u*dv u) (t+1))) := by
    simp only [Finset.mul_sum,← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro t ht
    rw [weighted_polynomial]
    ring
  unfold coordinateLoss lossVariation coordinateSecond coordinateThird
  rw [hs,weighted_polynomial]
  ring

/-- Reindexing the public loss by independent state coordinates. -/
theorem loss_coordinates (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (hasSeed : Bool) :
    loss c xs scale hasSeed=
    ∑ h∈Finset.range c.heads,∑ e∈Finset.range c.K,∑ j∈Finset.range c.V,
      coordinateLoss (headRate c h) (if hasSeed then xs.seed h e j else 0)
        (xs.finalIncoming h e j) scale (xs.q h · e) (xs.k h · e) (xs.v h · j)
        (xs.incoming h · j) c.T := by
  unfold loss output finalState state coordinateLoss
  simp only [Finset.sum_add_distrib]
  congr 1
  apply Finset.sum_congr rfl
  intro h hh
  rw [sum_time_first]
  apply Finset.sum_congr rfl
  intro t ht
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro j hj
  simp only [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro e he
  ring

def secondCoefficient (c : Shape) (xs : Inputs ℝ) (d : Variation) (scale : ℝ)
    (hasSeed : Bool) : ℝ :=
  ∑ h∈Finset.range c.heads,∑ e∈Finset.range c.K,∑ j∈Finset.range c.V,
    coordinateSecond (headRate c h) (if hasSeed then d.seed h e j else 0)
      (xs.finalIncoming h e j) scale (xs.q h · e) (xs.k h · e) (xs.v h · j)
      (xs.incoming h · j) (d.q h · e) (d.k h · e) (d.v h · j) c.T

def thirdCoefficient (c : Shape) (xs : Inputs ℝ) (d : Variation) (scale : ℝ) : ℝ :=
  ∑ h∈Finset.range c.heads,∑ e∈Finset.range c.K,∑ j∈Finset.range c.V,
    coordinateThird (headRate c h) scale (xs.incoming h · j)
      (d.q h · e) (d.k h · e) (d.v h · j) c.T

/-- Exact public loss expansion, with all coefficients independent of z.
This establishes fullVariation as the differential of the independent loss. -/
theorem loss_polynomial (c : Shape) (xs : Inputs ℝ) (d : Variation)
    (scale : ℝ) (hasSeed : Bool) (z : ℝ) :
    loss c (perturb xs d z) scale hasSeed=
      loss c xs scale hasSeed+z*fullVariation c xs d scale hasSeed+
      z^2*secondCoefficient c xs d scale hasSeed+z^3*thirdCoefficient c xs d scale := by
  rw [loss_coordinates,loss_coordinates]
  unfold fullVariation secondCoefficient thirdCoefficient
  simp only [Finset.mul_sum,← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro h hh
  apply Finset.sum_congr rfl
  intro e he
  apply Finset.sum_congr rfl
  intro j hj
  have seedEq : (if hasSeed then xs.seed h e j+z*d.seed h e j else 0)=
      (if hasSeed then xs.seed h e j else 0)+z*(if hasSeed then d.seed h e j else 0) := by
    cases hasSeed <;> simp
  simp only [perturb,seedEq]
  exact coordinate_polynomial _ _ _ _ _ _ _ _ _ _ _ _ _ _

/-- The computed gradient pairing is exactly the first coefficient of the
public loss polynomial, including terminal-state loss and seed variation. -/
theorem loss_gradient_polynomial (c : Shape) (xs : Inputs ℝ) (d : Variation)
    (scale : ℝ) (hasSeed : Bool) (z : ℝ) :
    loss c (perturb xs d z) scale hasSeed=
      loss c xs scale hasSeed+z*gradientPairing c xs d scale hasSeed+
      z^2*secondCoefficient c xs d scale hasSeed+z^3*thirdCoefficient c xs d scale := by
  rw [loss_polynomial,whole_vjp]

#axiomsClean weighted_polynomial
#axiomsClean coordinate_polynomial
#axiomsClean loss_coordinates
#axiomsClean loss_polynomial
#axiomsClean loss_gradient_polynomial
#specNonCircular perturb avoiding [fullVariation,gradientPairing]
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.MathTarget

