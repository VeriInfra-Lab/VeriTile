import bench.tritonbench_g.fused_recurrent_retention.Proofs.KernelInterface

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Logical
open Quoted
noncomputable section
set_option maxHeartbeats 1000000

/-- Equality only on live logical input coordinates and enabled optional
channels. No equality of unused memory or out-of-shape values is required. -/
structure Agreement (mode : Quoted.Mode) (s : Shape) (seed final : Bool) (x y : Inputs α) : Prop where
  q : ∀ h,h<s.BH → ∀ t,t<s.T → ∀ k,k<s.DK → x.q h t k=y.q h t k
  k : ∀ h,h<s.BH → ∀ t,t<s.T → ∀ k,k<s.DK → x.k h t k=y.k h t k
  v : ∀ h,h<s.BH → ∀ t,t<s.T → ∀ v,v<s.DV → x.v h t v=y.v h t v
  dout : mode=.backward → ∀ h,h<s.BH → ∀ t,t<s.T → ∀ v,v<s.DV → x.do_ h t v=y.do_ h t v
  initial : seed=Bool.true → ∀ h,h<s.BH → ∀ k,k<s.DK → ∀ v,v<s.DV → x.seed h k v=y.seed h k v
  finalGrad : mode=.backward → final=Bool.true → ∀ h,h<s.BH → ∀ k,k<s.DK → ∀ v,v<s.DV →
    x.finalGrad h k v=y.finalGrad h k v

variable {α : Type} {mode : Quoted.Mode} {s : Shape} {seed final : Bool} {x y : Inputs α}

theorem key_eq (a : Arithmetic α) (f g : Nat → Nat → Nat → α)
    (h t k : Nat) (eq : ∀ k,k<s.DK → f h t k=g h t k) :
    keyLoad a s f h t k=keyLoad a s g h t k := by
  unfold keyLoad
  split_ifs with hk
  · rw [eq k hk]
  · rfl

theorem value_eq (a : Arithmetic α) (f g : Nat → Nat → Nat → α)
    (h t v : Nat) (eq : ∀ v,v<s.DV → f h t v=g h t v) :
    valueLoad a s f h t v=valueLoad a s g h t v := by
  unfold valueLoad
  split_ifs with hv
  · rw [eq v hv]
  · rfl

theorem state_eq (a : Arithmetic α) (agree : Agreement mode s seed final x y)
    (h : Nat) (hh : h<s.BH) (k v n : Nat) (hn : n ≤ s.T) :
    state a s x seed h k v n=state a s y seed h k v n := by
  induction n with
  | zero =>
    simp only [state,foldN]
    cases hs : seed
    · rfl
    · simp only [hs,ite_true]
      unfold stateLoad
      split_ifs with hk
      · rw [agree.initial hs h hh k hk.1 v hk.2]
      · rfl
  | succ n ih =>
    change a.fma _ (state a s x seed h k v n) (a.mul (keyLoad a s x.k h n k) (valueLoad a s x.v h n v))=
      a.fma _ (state a s y seed h k v n) (a.mul (keyLoad a s y.k h n k) (valueLoad a s y.v h n v))
    rw [ih (by omega),key_eq a x.k y.k h n k (agree.k h hh n (by omega)),
      value_eq a x.v y.v h n v (agree.v h hh n (by omega))]

theorem reverse_eq (a : Arithmetic α) (agree : Agreement .backward s seed final x y)
    (h : Nat) (hh : h<s.BH) (k v n : Nat) (hn : n ≤ s.T) :
    reverseCarry a s x final h k v n=reverseCarry a s y final h k v n := by
  induction n with
  | zero =>
    simp only [reverseCarry,foldN]
    cases hf : final
    · rfl
    · simp only [hf,ite_true]
      unfold stateLoad
      split_ifs with hk
      · rw [agree.finalGrad rfl hf h hh k hk.1 v hk.2]
      · rfl
  | succ n ih =>
    change a.mul (a.fma (a.mul (keyLoad a s x.q h (s.T-1-n) k) (a.scale s.DK))
      (valueLoad a s x.do_ h (s.T-1-n) v) (reverseCarry a s x final h k v n)) _=
      a.mul (a.fma (a.mul (keyLoad a s y.q h (s.T-1-n) k) (a.scale s.DK))
      (valueLoad a s y.do_ h (s.T-1-n) v) (reverseCarry a s y final h k v n)) _
    rw [ih (by omega),key_eq a x.q y.q h _ k (agree.q h hh _ (by omega)),
      value_eq a x.do_ y.do_ h _ v (agree.dout rfl h hh _ (by omega))]

theorem adjoint_eq (a : Arithmetic α) (agree : Agreement .backward s seed final x y)
    (h : Nat) (hh : h<s.BH) (t : Nat) (ht : t<s.T) (k v : Nat) :
    adjoint a s x final h t k v=adjoint a s y final h t k v := by
  unfold adjoint
  rw [key_eq a x.q y.q h t k (agree.q h hh t ht),
    value_eq a x.do_ y.do_ h t v (agree.dout rfl h hh t ht),
    reverse_eq a agree h hh k v _ (by omega)]

theorem partialO_eq (a : Arithmetic α) (plans : Plans) (agree : Agreement mode s seed final x y)
    (h : Nat) (hh : h<s.BH) (t : Nat) (ht : t<s.T) (split v : Nat) :
    partialO a s plans x seed h t split v=partialO a s plans y seed h t split v := by
  unfold partialO
  congr 1
  · funext k
    exact state_eq a agree h hh _ v _ (by omega)
  · funext k
    rw [key_eq a x.q y.q h t _ (agree.q h hh t ht)]

theorem partialQ_eq (a : Arithmetic α) (plans : Plans) (agree : Agreement .backward s seed final x y)
    (h : Nat) (hh : h<s.BH) (t : Nat) (ht : t<s.T) (split k : Nat) :
    partialQ a s plans x seed h t split k=partialQ a s plans y seed h t split k := by
  unfold partialQ
  congr 2
  · funext v
    exact state_eq a agree h hh k _ _ (by omega)
  · funext v
    exact value_eq a x.do_ y.do_ h t _ (agree.dout rfl h hh t ht)

theorem partialK_eq (a : Arithmetic α) (plans : Plans) (agree : Agreement .backward s seed final x y)
    (h : Nat) (hh : h<s.BH) (t : Nat) (ht : t<s.T) (split k : Nat) :
    partialK a s plans x final h t split k=partialK a s plans y final h t split k := by
  unfold partialK
  congr 1
  · funext v
    exact adjoint_eq a agree h hh t ht k _
  · funext v
    exact value_eq a x.v y.v h t _ (agree.v h hh t ht)

theorem partialV_eq (a : Arithmetic α) (plans : Plans) (agree : Agreement .backward s seed final x y)
    (h : Nat) (hh : h<s.BH) (t : Nat) (ht : t<s.T) (split v : Nat) :
    partialV a s plans x final h t split v=partialV a s plans y final h t split v := by
  unfold partialV
  congr 1
  · funext k
    exact adjoint_eq a agree h hh t ht _ v
  · funext k
    exact key_eq a x.k y.k h t _ (agree.k h hh t ht)

#axiomsClean partialO_eq
#axiomsClean partialV_eq
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Logical
