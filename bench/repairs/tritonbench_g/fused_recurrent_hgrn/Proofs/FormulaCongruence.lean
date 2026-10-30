import bench.repairs.tritonbench_g.fused_recurrent_hgrn.IO

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn.FormulaCongruence
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type}

/-- Only the finite inputs consumed by this sequence matter. Optional state
inputs have no constraint when their source branch is disabled. -/
structure Agree (a b : Input α) (T bh d : Nat) (U V : Bool) : Prop where
  rows : ∀ r, r="x" ∨ r="g" ∨ r="do" → ∀ t, t<T → a r bh t d=b r bh t d
  h0 : U=Bool.true → a "h0" bh 0 d=b "h0" bh 0 d
  dht : V=Bool.true → a "dht" bh 0 d=b "dht" bh 0 d

theorem state_eq (p : Ops α) (a b : Input α) (T bh d : Nat) (U V : Bool)
    (h : Agree a b T bh d U V) (t : Nat) (ht : t≤T) :
    state p U a bh d t=state p U b bh d t := by
  induction t with
  | zero => cases U <;> simp [state,seed,h.h0]
  | succ t ih =>
    simp only [state,ih (by omega),h.rows "g" (by simp) t (by omega),
      h.rows "x" (by simp) t (by omega)]

theorem adjoint_eq (p : Ops α) (a b : Input α) (T bh d : Nat) (U V : Bool)
    (h : Agree a b T bh d U V) (n : Nat) (hn : n≤T) :
    adjoint p V a T bh d n=adjoint p V b T bh d n := by
  induction n with
  | zero => cases V <;> simp [adjoint,terminal,h.dht]
  | succ n ih =>
    simp only [adjoint,ih (by omega),h.rows "do" (by simp) (T-1-n) (by omega),
      h.rows "g" (by simp) (T-1-n) (by omega)]

theorem dx_eq (p : Ops α) (a b : Input α) (T bh d : Nat) (U V : Bool)
    (h : Agree a b T bh d U V) (t : Nat) (ht : t<T) :
    dx p V a T bh d t=dx p V b T bh d t := by
  simp only [dx,adjoint_eq p a b T bh d U V h (T-1-t) (by omega),
    h.rows "do" (by simp) t ht]

theorem dg_eq (p : Ops α) (a b : Input α) (T bh d : Nat) (U V : Bool)
    (h : Agree a b T bh d U V) (t : Nat) (ht : t<T) :
    dg p U V a T bh d t=dg p U V b T bh d t := by
  rw [dg,dg,dx_eq p a b T bh d U V h t ht]
  congr 1
  by_cases pos : 0<t
  · simpa only [previous,pos,if_true] using state_eq p a b T bh d U V h t (by omega)
  · cases U <;> simp [previous,pos,h.h0]

#axiomsClean state_eq
#axiomsClean dx_eq
#axiomsClean dg_eq
end
end VeriTile.Bench.Repairs.FusedRecurrentHgrn.FormulaCongruence
