import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Tactic
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.ChunkLinearAttn.Real
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 4000000

structure Inputs (T K V : Nat) where
  (q k : Fin T → Fin K → ℝ)
  v : Fin T → Fin V → ℝ
  initial : Fin K → Fin V → ℝ

def state {T K V : Nat} (x : Inputs T K V) (t : Fin T) (e : Fin K) (j : Fin V) : ℝ :=
  x.initial e j + ∑ u : Fin T, if u ≤ t then x.k u e*x.v u j else 0
def output {T K V : Nat} (scale : ℝ) (x : Inputs T K V) (t : Fin T) (j : Fin V) : ℝ :=
  scale * ∑ e : Fin K, x.q t e*state x t e j
def finalState {T K V : Nat} (x : Inputs T K V) (e : Fin K) (j : Fin V) : ℝ :=
  x.initial e j + ∑ u : Fin T, x.k u e*x.v u j
def adjoint {T K V : Nat} (scale : ℝ) (x : Inputs T K V)
    (G : Fin T → Fin V → ℝ) (R : Fin K → Fin V → ℝ) (u : Fin T) (e : Fin K) (j : Fin V) : ℝ :=
  R e j + scale * ∑ t : Fin T, if u ≤ t then x.q t e*G t j else 0
def gradients {T K V : Nat} (scale : ℝ) (x : Inputs T K V)
    (G : Fin T → Fin V → ℝ) (R : Fin K → Fin V → ℝ) : Inputs T K V where
  q := fun t e => scale * ∑ j : Fin V, state x t e j*G t j
  k := fun u e => ∑ j : Fin V, adjoint scale x G R u e j*x.v u j
  v := fun u j => ∑ e : Fin K, adjoint scale x G R u e j*x.k u e
  initial := fun e j => R e j + scale * ∑ t : Fin T, x.q t e*G t j

def perturb {T K V : Nat} (x dx : Inputs T K V) (ε : ℝ) : Inputs T K V where
  q := fun t e => x.q t e+ε*dx.q t e
  k := fun t e => x.k t e+ε*dx.k t e
  v := fun t j => x.v t j+ε*dx.v t j
  initial := fun e j => x.initial e j+ε*dx.initial e j
def stateVariation {T K V : Nat} (x dx : Inputs T K V) (t : Fin T) (e : Fin K) (j : Fin V) : ℝ :=
  dx.initial e j + ∑ u : Fin T, if u ≤ t then dx.k u e*x.v u j+x.k u e*dx.v u j else 0
def finalVariation {T K V : Nat} (x dx : Inputs T K V) (e : Fin K) (j : Fin V) : ℝ :=
  dx.initial e j + ∑ u : Fin T, (dx.k u e*x.v u j+x.k u e*dx.v u j)
def outputVariation {T K V : Nat} (scale : ℝ) (x dx : Inputs T K V) (t : Fin T) (j : Fin V) : ℝ :=
  scale * ∑ e : Fin K, (dx.q t e*state x t e j+x.q t e*stateVariation x dx t e j)
def loss {T K V : Nat} (scale : ℝ) (x : Inputs T K V)
    (G : Fin T → Fin V → ℝ) (R : Fin K → Fin V → ℝ) : ℝ :=
  (∑ t : Fin T, ∑ j : Fin V, G t j*output scale x t j) +
  (∑ e : Fin K, ∑ j : Fin V, R e j*finalState x e j)
def lossVariation {T K V : Nat} (scale : ℝ) (x dx : Inputs T K V)
    (G : Fin T → Fin V → ℝ) (R : Fin K → Fin V → ℝ) : ℝ :=
  (∑ t : Fin T, ∑ j : Fin V, G t j*outputVariation scale x dx t j) +
  (∑ e : Fin K, ∑ j : Fin V, R e j*finalVariation x dx e j)
def pairing {T K V : Nat} (dx grad : Inputs T K V) : ℝ :=
  (∑ t : Fin T, ∑ e : Fin K, dx.q t e*grad.q t e) +
  (∑ t : Fin T, ∑ e : Fin K, dx.k t e*grad.k t e) +
  (∑ t : Fin T, ∑ j : Fin V, dx.v t j*grad.v t j) +
  (∑ e : Fin K, ∑ j : Fin V, dx.initial e j*grad.initial e j)

theorem affine_deriv (a b : ℝ) : HasDerivAt (fun ε : ℝ => a+ε*b) b 0 := by
  simpa using ((hasDerivAt_id (0 : ℝ)).mul_const b).const_add a

theorem state_deriv {T K V : Nat} (x dx : Inputs T K V) (t : Fin T) (e : Fin K) (j : Fin V) :
    HasDerivAt (fun ε => state (perturb x dx ε) t e j) (stateVariation x dx t e j) 0 := by
  apply HasDerivAt.add (affine_deriv _ _)
  apply HasDerivAt.fun_sum
  intro u _
  by_cases h : u ≤ t
  · simpa only [h,if_true,perturb,zero_mul,add_zero] using
      (affine_deriv (x.k u e) (dx.k u e)).mul (affine_deriv (x.v u j) (dx.v u j))
  · simpa only [h,if_false] using hasDerivAt_const (0 : ℝ) (0 : ℝ)

theorem final_deriv {T K V : Nat} (x dx : Inputs T K V) (e : Fin K) (j : Fin V) :
    HasDerivAt (fun ε => finalState (perturb x dx ε) e j) (finalVariation x dx e j) 0 := by
  apply HasDerivAt.add (affine_deriv _ _)
  apply HasDerivAt.fun_sum
  intro u _
  simpa only [perturb,zero_mul,add_zero] using
    (affine_deriv (x.k u e) (dx.k u e)).mul (affine_deriv (x.v u j) (dx.v u j))

theorem output_deriv {T K V : Nat} (scale : ℝ) (x dx : Inputs T K V) (t : Fin T) (j : Fin V) :
    HasDerivAt (fun ε => output scale (perturb x dx ε) t j) (outputVariation scale x dx t j) 0 := by
  apply HasDerivAt.const_mul
  apply HasDerivAt.fun_sum
  intro e _
  simpa only [perturb,state,zero_mul,add_zero] using
    (affine_deriv (x.q t e) (dx.q t e)).mul (state_deriv x dx t e j)

theorem loss_deriv {T K V : Nat} (scale : ℝ) (x dx : Inputs T K V)
    (G : Fin T → Fin V → ℝ) (R : Fin K → Fin V → ℝ) :
    HasDerivAt (fun ε => loss scale (perturb x dx ε) G R) (lossVariation scale x dx G R) 0 := by
  apply HasDerivAt.add
  · apply HasDerivAt.fun_sum
    intro t _
    apply HasDerivAt.fun_sum
    intro j _
    exact (output_deriv scale x dx t j).const_mul (G t j)
  · apply HasDerivAt.fun_sum
    intro e _
    apply HasDerivAt.fun_sum
    intro j _
    exact (final_deriv x dx e j).const_mul (R e j)

#axiomsClean state_deriv
#axiomsClean final_deriv
#axiomsClean output_deriv
#axiomsClean loss_deriv
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.Real
