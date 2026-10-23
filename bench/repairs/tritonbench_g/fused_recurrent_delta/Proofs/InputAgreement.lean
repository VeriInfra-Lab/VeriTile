import bench.repairs.tritonbench_g.fused_recurrent_delta.IO
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.InputAgreement
open VeriTile.Triton FP.Structural KernelRegisters KernelBackwardRegisters Interface
noncomputable section
variable {α : Type} [Inhabited α]

/-- Only time coordinates actually used by the complete kernel are related.
The old unconstrained values outside the finite sequence remain unrestricted. -/
structure Forward {s : Shape} (a b : KernelReference.ForwardInput s α) : Prop where
  key : ∀ t,t<s.T → a.key t=b.key t
  query : ∀ t,t<s.T → a.query t=b.query t
  value : ∀ t,t<s.T → a.value t=b.value t
  beta : ∀ t,t<s.T → a.beta t=b.beta t
  seed : a.seed=b.seed

namespace Forward
variable (M : Algebra α) (s : Shape) {a b : KernelReference.ForwardInput s α}
    (h : Forward a b)
include h
theorem carry (n : Nat) (hn : n≤s.T) :
    KernelReference.forward M s a n=KernelReference.forward M s b n := by
  induction n with
  | zero => exact h.seed
  | succ n ih =>
    have ht : n<s.T := by omega
    simp only [KernelReference.forward,ih (by omega),h.key n ht,h.value n ht,h.beta n ht]
theorem residual (t : Nat) (ht : t<s.T) :
    KernelReference.residual M s a t=KernelReference.residual M s b t := by
  funext j
  simp only [KernelReference.residual,h.carry M s t (by omega),h.key t ht,h.value t ht]
theorem output (scale : ℝ) (t : Nat) (ht : t<s.T) :
    KernelReference.output M s a scale t=KernelReference.output M s b scale t := by
  funext j
  simp only [KernelReference.output,h.carry M s (t+1) (by omega),h.query t ht]
end Forward

structure Backward {s : Shape} (a b : KernelBackwardReference.Input s α) : Prop where
  key : ∀ t,t<s.T → a.key t=b.key t
  query : ∀ t,t<s.T → a.query t=b.query t
  value : ∀ t,t<s.T → a.value t=b.value t
  beta : ∀ t,t<s.T → a.beta t=b.beta t
  incoming : ∀ t,t<s.T → a.incoming t=b.incoming t
  initial : a.initial=b.initial
  terminal : a.terminal=b.terminal

namespace Backward
variable (M : Algebra α) (s : Shape) {a b : KernelBackwardReference.Input s α}
    (h : Backward a b)
include h
theorem forward (n : Nat) (hn : n≤s.T) :
    KernelBackwardReference.forward M s a n=KernelBackwardReference.forward M s b n := by
  induction n with
  | zero => exact h.initial
  | succ n ih =>
    have ht : n<s.T := by omega
    simp only [KernelBackwardReference.forward,ih (by omega),h.key n ht,h.value n ht,h.beta n ht]
theorem reverse (scale : ℝ) (n : Nat) (hn : n≤s.T) :
    KernelBackwardReference.reverse M s scale a n=KernelBackwardReference.reverse M s scale b n := by
  induction n with
  | zero => exact h.terminal
  | succ n ih =>
    have ht : s.T-1-n<s.T := by omega
    funext ij
    simp only [KernelBackwardReference.reverse,KernelBackwardReference.retreat,
      KernelBackwardReference.enhance,KernelBackwardReference.dv,KernelBackwardReference.rawDv,
      ih (by omega),h.key _ ht,h.query _ ht,h.incoming _ ht,h.beta _ ht]
theorem enhanced (scale : ℝ) (t : Nat) (ht : t<s.T) :
    Interface.enhanced M s scale a t=Interface.enhanced M s scale b t := by
  funext ij
  simp only [Interface.enhanced,KernelBackwardReference.enhance,
    h.reverse M s scale (s.T-1-t) (by omega),h.query t ht,h.incoming t ht]
theorem query_value (scale : ℝ) (t : Nat) (ht : t<s.T) :
    backwardQuery M s scale a t=backwardQuery M s scale b t := by
  funext i
  simp only [backwardQuery,KernelBackwardReference.dq,
    h.forward M s (t+1) (by omega),h.incoming t ht]
theorem value_value (scale : ℝ) (t : Nat) (ht : t<s.T) :
    KernelBackwardReference.dv M s a t (Interface.enhanced M s scale a t)=
      KernelBackwardReference.dv M s b t (Interface.enhanced M s scale b t) := by
  funext j
  simp only [KernelBackwardReference.dv,KernelBackwardReference.rawDv,
    h.enhanced M s scale t ht,h.key t ht,h.beta t ht]
theorem beta_value (scale : ℝ) (t : Nat) (ht : t<s.T) :
    KernelBackwardReference.dbeta M s a t (Interface.enhanced M s scale a t)=
      KernelBackwardReference.dbeta M s b t (Interface.enhanced M s scale b t) := by
  funext j
  simp only [KernelBackwardReference.dbeta,KernelBackwardReference.rawDv,
    h.enhanced M s scale t ht,h.key t ht,h.value t ht]
theorem key_value (scale : ℝ) (iv t : Nat) (ht : t<s.T) :
    backwardKey M s scale iv a t=backwardKey M s scale iv b t := by
  funext i
  simp only [backwardKey,KernelBackwardReference.dk,KernelBackwardReference.dv,
    KernelBackwardReference.rawDv,h.enhanced M s scale t ht,
    h.forward M s t (by omega),h.key t ht,h.value t ht,h.beta t ht]
end Backward
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.InputAgreement
