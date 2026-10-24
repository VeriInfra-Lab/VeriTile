import bench.repairs.tritonbench_g.chunk_linear_attn.RealTarget
namespace VeriTile.Bench.Repairs.ChunkLinearAttn.Real
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def zeroInputs (T K V : Nat) : Inputs T K V := ⟨fun _ _ => 0,fun _ _ => 0,fun _ _ => 0,fun _ _ => 0⟩
def basisQ {T K V : Nat} (t : Fin T) (e : Fin K) : Inputs T K V :=
  {zeroInputs T K V with q:=fun u k => if u=t then if k=e then 1 else 0 else 0}
def basisK {T K V : Nat} (t : Fin T) (e : Fin K) : Inputs T K V :=
  {zeroInputs T K V with k:=fun u k => if u=t then if k=e then 1 else 0 else 0}
def basisV {T K V : Nat} (t : Fin T) (j : Fin V) : Inputs T K V :=
  {zeroInputs T K V with v:=fun u v => if u=t then if v=j then 1 else 0 else 0}
def basisInitial {T K V : Nat} (e : Fin K) (j : Fin V) : Inputs T K V :=
  {zeroInputs T K V with initial:=fun k v => if k=e then if v=j then 1 else 0 else 0}

theorem masked_select {ι : Type} [Fintype ι] [DecidableEq ι] (P : ι → Prop) [DecidablePred P]
    (target : ι) (f : ι → ℝ) :
    (∑ i : ι, if P i then (if i=target then f i else 0) else 0)=
      (if P target then f target else 0) := by
  calc
    _ = ∑ i : ι, if i=target then (if P i then f i else 0) else 0 := by
      apply Finset.sum_congr rfl
      intro i _
      by_cases hp : P i <;> by_cases hi : i=target <;> simp [hp,hi]
    _ = _ := by simp

theorem query_variation {T K V : Nat} (scale : ℝ) (x : Inputs T K V)
    (G : Fin T → Fin V → ℝ) (R : Fin K → Fin V → ℝ) (t : Fin T) (e : Fin K) :
    lossVariation scale x (basisQ t e) G R=(gradients scale x G R).q t e := by
  simp [lossVariation,outputVariation,stateVariation,finalVariation,basisQ,zeroInputs,gradients,
    mul_ite,ite_mul,Finset.mul_sum,Finset.sum_mul,mul_comm,mul_left_comm,mul_assoc]

theorem initial_variation {T K V : Nat} (scale : ℝ) (x : Inputs T K V)
    (G : Fin T → Fin V → ℝ) (R : Fin K → Fin V → ℝ) (e : Fin K) (j : Fin V) :
    lossVariation scale x (basisInitial e j) G R=(gradients scale x G R).initial e j := by
  simp [lossVariation,outputVariation,stateVariation,finalVariation,basisInitial,zeroInputs,gradients,
    mul_ite,ite_mul,Finset.mul_sum,Finset.sum_mul,mul_comm,mul_left_comm,mul_assoc,add_comm]

theorem key_variation {T K V : Nat} (scale : ℝ) (x : Inputs T K V)
    (G : Fin T → Fin V → ℝ) (R : Fin K → Fin V → ℝ) (u : Fin T) (e : Fin K) :
    lossVariation scale x (basisK u e) G R=(gradients scale x G R).k u e := by
  simp [lossVariation,outputVariation,stateVariation,finalVariation,basisK,zeroInputs,gradients,adjoint,
    mul_ite,ite_mul,Finset.mul_sum,Finset.sum_mul,mul_comm,mul_left_comm,mul_assoc,
    Finset.sum_add_distrib,add_mul,mul_add,add_comm,masked_select,Finset.sum_ite_irrel]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro t _
  by_cases h : u ≤ t <;> simp [h]

theorem value_variation {T K V : Nat} (scale : ℝ) (x : Inputs T K V)
    (G : Fin T → Fin V → ℝ) (R : Fin K → Fin V → ℝ) (u : Fin T) (j : Fin V) :
    lossVariation scale x (basisV u j) G R=(gradients scale x G R).v u j := by
  simp [lossVariation,outputVariation,stateVariation,finalVariation,basisV,zeroInputs,gradients,adjoint,
    mul_ite,ite_mul,Finset.mul_sum,Finset.sum_mul,mul_comm,mul_left_comm,mul_assoc,
    Finset.sum_add_distrib,add_mul,mul_add,add_comm,masked_select,Finset.sum_ite_irrel]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro t _
  by_cases h : u ≤ t <;> simp [h]

/-- Complete VJP: every coordinate of every differentiable input is the actual
derivative of the two-output cotangent pairing. No autograd result is assumed. -/
structure IsVJP {T K V : Nat} (scale : ℝ) (x : Inputs T K V)
    (G : Fin T → Fin V → ℝ) (R : Fin K → Fin V → ℝ) (grad : Inputs T K V) : Prop where
  query : ∀ t e, HasDerivAt (fun ε => loss scale (perturb x (basisQ t e) ε) G R) (grad.q t e) 0
  key : ∀ t e, HasDerivAt (fun ε => loss scale (perturb x (basisK t e) ε) G R) (grad.k t e) 0
  value : ∀ t j, HasDerivAt (fun ε => loss scale (perturb x (basisV t j) ε) G R) (grad.v t j) 0
  initial : ∀ e j, HasDerivAt (fun ε => loss scale (perturb x (basisInitial e j) ε) G R) (grad.initial e j) 0

theorem gradients_correct {T K V : Nat} (scale : ℝ) (x : Inputs T K V)
    (G : Fin T → Fin V → ℝ) (R : Fin K → Fin V → ℝ) : IsVJP scale x G R (gradients scale x G R) := by
  constructor
  · intro t e
    simpa only [query_variation] using loss_deriv scale x (basisQ t e) G R
  · intro t e
    simpa only [key_variation] using loss_deriv scale x (basisK t e) G R
  · intro t j
    simpa only [value_variation] using loss_deriv scale x (basisV t j) G R
  · intro e j
    simpa only [initial_variation] using loss_deriv scale x (basisInitial e j) G R

#axiomsClean gradients_correct
#specNonCircular IsVJP avoiding [gradients]
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.Real
