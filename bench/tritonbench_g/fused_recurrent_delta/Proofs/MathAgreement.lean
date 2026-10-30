import bench.tritonbench_g.fused_recurrent_delta.Proofs.Mathematics
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.MathAgreement
noncomputable section

structure Agree {K : Nat} (T : Nat) (a b : Math.Input K) : Prop where
  q : ∀ t,t<T → a.q t=b.q t
  k : ∀ t,t<T → a.k t=b.k t
  x : ∀ t,t<T → a.x t=b.x t
  beta : ∀ t,t<T → a.beta t=b.beta t
  h0 : a.h0=b.h0

namespace Agree
variable {K T : Nat} {a b : Math.Input K} (h : Agree T a b)
include h

theorem state (n : Nat) (hn : n≤T) : Math.state a n=Math.state b n := by
  induction n with
  | zero => exact h.h0
  | succ n ih =>
    have ht : n<T := by omega
    simp only [Math.state,ih (by omega),h.k n ht,h.x n ht,h.beta n ht]
theorem output (scale : ℝ) (t : Nat) (ht : t<T) : Math.output scale a t=Math.output scale b t := by
  simp only [Math.output,h.state (t+1) (by omega),h.q t ht]
theorem residual (t : Nat) (ht : t<T) :
    Math.residual (Math.state a t) (a.k t) (a.x t)=Math.residual (Math.state b t) (b.k t) (b.x t) := by
  rw [h.state t (by omega),h.k t ht,h.x t ht]
theorem reverse (scale : ℝ) (g g' : Nat → ℝ) (hg : ∀ t,t<T → g t=g' t)
    (terminal : Math.Vec K) (n : Nat) (hn : n≤T) :
    Math.reverse scale a g terminal T n=Math.reverse scale b g' terminal T n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    have ht : T-1-n<T := by omega
    simp only [Math.reverse,ih (by omega),h.state (T-1-n) (by omega),h.q _ ht,h.k _ ht,h.x _ ht,h.beta _ ht,hg _ ht]
theorem gradient (scale : ℝ) (g g' : Nat → ℝ) (hg : ∀ t,t<T → g t=g' t)
    (terminal : Math.Vec K) (t : Nat) (ht : t<T) :
    Math.gradient scale a g terminal T t=Math.gradient scale b g' terminal T t := by
  simp only [Math.gradient,Math.carry,h.state t (by omega),h.q t ht,h.k t ht,h.x t ht,h.beta t ht,hg t ht,
    h.reverse scale g g' hg terminal (T-(t+1)) (by omega)]
end Agree

/-- Internal finite extension for reusing the existing all-time algebraic law;
it does not constrain the caller's values at any unused time. -/
def cutoff {K : Nat} (T : Nat) (a : Math.Input K) : Math.Input K where
  q := fun t => if t<T then a.q t else fun _ => 0
  k := fun t => if t<T then a.k t else fun _ => 0
  x := fun t => if t<T then a.x t else 0
  beta := fun t => if t<T then a.beta t else 0
  h0 := a.h0

theorem cutoff_agrees {K : Nat} (T : Nat) (a : Math.Input K) : Agree T (cutoff T a) a := by
  constructor
  · intro t ht; simp [cutoff,ht]
  · intro t ht; simp [cutoff,ht]
  · intro t ht; simp [cutoff,ht]
  · intro t ht; simp [cutoff,ht]
  · rfl

#axiomsClean Agree.state
#axiomsClean Agree.gradient
#axiomsClean cutoff_agrees
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.MathAgreement
