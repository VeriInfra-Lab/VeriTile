import Mathlib.Tactic

namespace VeriTile.Bench.Repairs.ContextAttnMistral.OutputView
noncomputable section

/-- Ordinary tensor copy contract, independent of Torch internals. Coordinates
include every cell of supplied o. `encode` writes its physical representation
and `decode` reads its logical value. The view may overlap at inactive cells.
Lazy negation is an involution; FP sign-bit toggling fits the generic law. -/
structure View (Coord Value : Type) where
  addr : Coord → Nat
  encode : Value → Value
  decode : Value → Value
  decode_encode : ∀ x, decode (encode x) = x
  encode_decode : ∀ x, encode (decode x) = x

def View.read {C V : Type} (view : View C V) (memory : Nat → V) (i : C) : V :=
  view.decode (memory (view.addr i))

/-- Only simultaneously active output coordinates must be nonoverlapping. -/
def View.Writable {C V : Type} (view : View C V) (active : C → Prop) : Prop :=
  ∀ i j, active i → active j → view.addr i = view.addr j → i = j

/-- The wrapper copies only the covered slices, after snapshotting metadata.
Coordinates outside those slices do not participate in Torch copy_. -/
def View.copy {C V : Type} (view : View C V) (active : C → Prop)
    (memory : Nat → V) (value : C → V) : Nat → V := by
  classical
  exact fun a => if h : ∃ i, active i ∧ view.addr i = a then
    view.encode (value (Classical.choose h)) else memory a

theorem copy_at {C V : Type} (view : View C V) (active : C → Prop)
    (hw : view.Writable active) (memory : Nat → V) (value : C → V) (i : C) (hi : active i) :
    view.read (view.copy active memory value) i = value i := by
  classical
  unfold View.read View.copy
  rw [dif_pos ⟨i,hi,rfl⟩]
  let h : ∃ j, active j ∧ view.addr j = view.addr i := ⟨i,hi,rfl⟩
  have hj : Classical.choose h = i :=
    hw _ _ (Classical.choose_spec h).1 hi (Classical.choose_spec h).2
  change view.decode (view.encode (value (Classical.choose h))) = value i
  rw [hj,view.decode_encode]

/-- Full original-storage frame, including gaps and all addresses outside
active supplied-output writes. Inactive coordinates may alias active storage. -/
theorem copy_frame {C V : Type} (view : View C V) (active : C → Prop)
    (memory : Nat → V) (value : C → V) (a : Nat)
    (ha : ¬ ∃ i, active i ∧ view.addr i = a) :
    view.copy active memory value a = memory a := by
  classical
  simp [View.copy,ha]

/-- Both ordinary and negative output views satisfy the logical copy laws. -/
def realView {C : Type} (addr : C → Nat) (negative : Bool) : View C ℝ where
  addr := addr
  encode := if negative then Neg.neg else id
  decode := if negative then Neg.neg else id
  decode_encode := by intro x; cases negative <;> simp
  encode_decode := by intro x; cases negative <;> simp
end
end VeriTile.Bench.Repairs.ContextAttnMistral.OutputView
