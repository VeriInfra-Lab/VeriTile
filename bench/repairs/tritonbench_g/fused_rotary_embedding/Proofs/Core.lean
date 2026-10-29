import Mathlib.Tactic
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.FusedRotaryEmbedding

inductive Region where
  | q | k | v | cos | sin | kcache | vcache | lengths | table | other (id : Nat)
  deriving DecidableEq, Repr

abbrev Memory (α : Type) := Region → Nat → α
def component {α : Type} (p : α × α) (second : Bool) : α := if second then p.2 else p.1

namespace Stores
variable {α : Type}
abbrev Write (α : Type) := (Region × Nat) × α
def store (m : Memory α) (w : Write α) : Memory α :=
  fun r a => if (r,a)=w.1 then w.2 else m r a
def commit (ws : List (Write α)) (m : Memory α) : Memory α := ws.foldl store m
def Coherent (ws : List (Write α)) : Prop := ∀ a∈ws,∀ b∈ws,a.1=b.1 → a.2=b.2

theorem frame (ws : List (Write α)) (m : Memory α) (r : Region) (a : Nat)
    (outside : ∀ w∈ws,(r,a)≠w.1) : commit ws m r a=m r a := by
  induction ws generalizing m with
  | nil => rfl
  | cons w ws ih =>
    change commit ws (store m w) r a=m r a
    rw [ih _ (fun x hx => outside x (by simp [hx]))]
    simp [store,outside w (by simp)]

theorem stable (ws : List (Write α)) (m : Memory α) (r : Region) (a : Nat) (value : α)
    (old : m r a=value) (agree : ∀ w∈ws,w.1=(r,a) → w.2=value) : commit ws m r a=value := by
  induction ws generalizing m with
  | nil => exact old
  | cons w ws ih =>
    apply ih (store m w)
    · by_cases eq : (r,a)=w.1
      · simp [store,eq,agree w (by simp) eq.symm]
      · simp [store,eq,old]
    · exact fun x hx eq => agree x (by simp [hx]) eq

theorem readback (ws : List (Write α)) (m : Memory α) (good : Coherent ws)
    (w : Write α) (member : w∈ws) : commit ws m w.1.1 w.1.2=w.2 := by
  induction ws generalizing m with
  | nil => simp at member
  | cons x ws ih =>
    change commit ws (store m x) w.1.1 w.1.2=w.2
    rcases List.mem_cons.mp member with eq|tail
    · subst w
      apply stable
      · simp [store]
      · intro v hv same
        exact good v (by simp [hv]) x (by simp) same
    · exact ih _ (fun a ha b hb eq => good a (by simp [ha]) b (by simp [hb]) eq) tail
end Stores
end VeriTile.Bench.Repairs.FusedRotaryEmbedding
