import VeriTile.Triton
import VeriTile.Triton.Semantics.MaskedReduction
import VeriTile.Meta.StatementAudit

/-! Independent scalar formulas for the complete `_softmax_backward`.
The causal read mask and the less restrictive output mask are distinct.
In particular, the LOG branch writes a negative row sum at future causal
lanes, because its masked saved output is zero and exp(0) is one.
-/
namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward
open VeriTile.Triton
open scoped BigOperators

def readActive (K row : Nat) (causal : Bool) (lane : Nat) : Bool :=
  decide (lane < K) && (!causal || decide (lane ≤ row))

noncomputable def masked {D : Nat} (K row : Nat) (causal : Bool)
    (xs : Fin D → ℝ) (lane : Fin D) : ℝ :=
  if readActive K row causal lane.val then xs lane else 0

/-- Pure finite sums over input samples; no interpreter or tile operation. -/
noncomputable def formula {D : Nat} (K row : Nat) (log causal : Bool)
    (gradient saved : Fin D → ℝ) (lane : Fin D) : ℝ :=
  let g := masked K row causal gradient
  let o := masked K row causal saved
  if log then g lane - Real.exp (o lane) * ∑ j, g j
  else o lane * (g lane - ∑ j, g j * o j)

theorem masked_congr {D : Nat} (K row : Nat) (causal : Bool)
    (xs ys : Fin D → ℝ)
    (h : ∀ j, readActive K row causal j.val = Bool.true → xs j = ys j) :
    masked K row causal xs = masked K row causal ys := by
  funext j
  by_cases hj : readActive K row causal j.val = Bool.true
  · simp only [masked, hj, if_true, h j hj]
  · simp only [masked, hj, Bool.false_eq_true, if_false]

theorem formula_congr {D : Nat} (K row : Nat) (log causal : Bool)
    (g g' o o' : Fin D → ℝ)
    (hg : ∀ j, readActive K row causal j.val = Bool.true → g j = g' j)
    (ho : ∀ j, readActive K row causal j.val = Bool.true → o j = o' j)
    (i : Fin D) : formula K row log causal g o i = formula K row log causal g' o' i := by
  simp only [formula, masked_congr K row causal g g' hg, masked_congr K row causal o o' ho]

/-- This is the original LOG/CAUSAL behavior, not a zero-output assumption. -/
theorem causal_log_future {D : Nat} (K row : Nat) (g o : Fin D → ℝ)
    (i : Fin D) (hi : row < i.val) :
    formula K row Bool.true Bool.true g o i = -∑ j, masked K row Bool.true g j := by
  have hn : ¬i.val ≤ row := Nat.not_le.mpr hi
  simp [formula, masked, readActive, hn]

theorem causal_plain_future {D : Nat} (K row : Nat) (g o : Fin D → ℝ)
    (i : Fin D) (hi : row < i.val) :
    formula K row Bool.false Bool.true g o i = 0 := by
  have hn : ¬i.val ≤ row := Nat.not_le.mpr hi
  simp [formula, masked, readActive, hn]

noncomputable def inputTile (K row D : Nat) (causal : Bool)
    (xs : Fin D → ℝ) : Tile .real [D] :=
  ⟨fun idx => some (masked K row causal xs idx.1)⟩

/-- A semantic bridge expression, separate from the public scalar formula. -/
noncomputable def resultTile (D : Nat) (log : Bool) (g o : Tile .real [D]) : Tile .real [D] :=
  if log then
    let s := Tile.reduceSum (shape := [D]) ⟨0, by simp⟩ Bool.false g
    Tile.bop (NumericDType.sub .real) (.consSame .nil) g
      (Tile.bop (NumericDType.mul .real) .scalarR (Tile.uop WithBot.realExp o) s)
  else
    let s := Tile.reduceSum (shape := [D]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (.consSame .nil) g o)
    Tile.bop (NumericDType.mul .real) (.consSame .nil) o
      (Tile.bop (NumericDType.sub .real) .scalarR g s)

theorem resultTile_formula (K row D : Nat) (log causal : Bool)
    (g o : Fin D → ℝ) (i : Fin D) :
    (resultTile D log (inputTile K row D causal g) (inputTile K row D causal o)).data
      (i, PUnit.unit) = some (formula K row log causal g o i) := by
  cases log <;>
    simp [resultTile, inputTile, formula, Tile.bop, Tile.uop,
      Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
      TileShape.eraseAxis, TileShape.insertAxisIndex,
      NumericDType.mul, NumericDType.sub, Broadcast.leftIndex,
      Broadcast.rightIndex, WithBot.sum_someTerm_eq_some]
  all_goals rfl

theorem causal_log_nonzero_example :
    formula 2 0 Bool.true Bool.true (fun _ : Fin 4 => (2 : ℝ))
      (fun _ : Fin 4 => (0 : ℝ)) ⟨1, by decide⟩ = -2 := by
  norm_num [formula, masked, readActive, Fin.sum_univ_succ]

#axiomsClean masked_congr
#axiomsClean formula_congr
#axiomsClean causal_log_future
#axiomsClean causal_plain_future
#axiomsClean resultTile_formula
#axiomsClean causal_log_nonzero_example
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward
