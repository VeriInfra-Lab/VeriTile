import VeriTile.Triton.Core.Shape
import Mathlib.Data.Finset.Lattice.Fold
import Mathlib.Data.Fintype.Fin
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-! Exact discrete reductions shared by real and FP execution. Numerical
floating algebras supply no assumption or operation for these control values. -/
namespace VeriTile.Triton.TileShape
noncomputable section

/-- Triton promotes a bool sum to uint32. Reduction order does not affect
modular integer addition, so counting true lanes then reducing modulo 2^32
retains the result even when an abstract axis is larger than uint32. -/
def boolCount32 {n : Nat} (flags : Fin n → Bool) : Nat :=
  (∑ i, if flags i then (1 : Nat) else 0) % 2^32

def discreteBoolSum {shape : TileShape} (axis : Fin shape.length) :
    (keepDims : Bool) → (TileIndex shape → Bool) →
      TileIndex (reduceShape shape axis keepDims) → Nat
  | Bool.false, x => fun out => boolCount32 (fun k => x (insertAxisIndex shape axis out k))
  | Bool.true, x => fun out => boolCount32 (fun k => x (replaceAxisIndex shape axis out k))

/-- Exact integer argmax. The ordered fold replaces the current best only
on strict improvement, retaining the leftmost lane among equal maxima.
An empty axis retains the legacy real interpreter's index-zero convention. -/
def leftmostIntArgMax {n : Nat} (values : Fin n → Int) : Nat :=
  if h : 0 < n then
    ((List.finRange n).foldl
      (fun best candidate => if values best < values candidate then candidate else best)
      ⟨0,h⟩).val
  else 0

def discreteIntArgMax {shape : TileShape} (axis : Fin shape.length)
    (x : TileIndex shape → Int) : TileIndex (eraseAxis shape axis) → Nat :=
  fun out => leftmostIntArgMax (fun k => x (insertAxisIndex shape axis out k))

/-- Exact maximum for discrete carriers. A zero-sized reduced axis fails;
no zero initializer is inserted for a nonempty signed tile. -/
def discreteMaximum {β : Type} [SemilatticeSup β] {shape : TileShape}
    (axis : Fin shape.length) : (keepDims : Bool) → (TileIndex shape → β) →
      Option (TileIndex (reduceShape shape axis keepDims) → β)
  | Bool.false, x =>
    if h : 0 < axisDim shape axis then
      some (fun out => (Finset.univ : Finset (Fin (axisDim shape axis))).sup'
        ⟨⟨0,h⟩, Finset.mem_univ _⟩ (fun k => x (insertAxisIndex shape axis out k)))
    else none
  | Bool.true, x =>
    if h : 0 < axisDim shape axis then
      some (fun out => (Finset.univ : Finset (Fin (axisDim shape axis))).sup'
        ⟨⟨0,h⟩, Finset.mem_univ _⟩ (fun k => x (replaceAxisIndex shape axis out k)))
    else none

end
end VeriTile.Triton.TileShape
