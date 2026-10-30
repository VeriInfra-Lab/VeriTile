import VeriTile.Triton
import VeriTile.Meta.StatementAudit

/-! Independent finite mathematics for every admitted forward mask mode and flag.
The extended result retains the LOG/CAUSAL negative-infinity lane; the real
observation is explicitly separate because the algorithm's store demotes it.
-/
namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward
open scoped BigOperators

inductive MaskMode where
  | absent | qk | bk
  deriving DecidableEq, Repr

def MaskMode.python : MaskMode → Option String
  | .absent => none
  | .qk => some "qk"
  | .bk => some "bk"

def MaskMode.present (mode : MaskMode) : Bool := mode != .absent
def MaskMode.isQK (mode : MaskMode) : Bool := mode == .qk
def MaskMode.isBK (mode : MaskMode) : Bool := mode == .bk

theorem mask_conditions (mode : MaskMode) :
    mode.present = mode.python.isSome ∧
    mode.isQK = (mode.python == some "qk") ∧
    mode.isBK = (mode.python == some "bk") := by
  cases mode <;> decide

def readActive (K row : Nat) (causal : Bool) (lane : Nat) : Bool :=
  decide (lane < K) && (!causal || decide (lane ≤ row))

/-- This is exactly the source's qk/bk address convention. In particular,
`stride_m` is the wrapper's M.stride(-1), not an inferred row stride. -/
def maskAddress (mode : MaskMode) (batch row stride lane : Nat) : Nat :=
  (if mode = .qk then row else batch) * stride + lane

noncomputable def score {D : Nat} (K row : Nat) (causal : Bool)
    (mode : MaskMode) (xs masks : Fin D → ℝ) (lane : Fin D) : WithBot ℝ :=
  if readActive K row causal lane.val then
    some (xs lane + if mode.present then masks lane else 0)
  else none

/-- Supremum over the full padded tile, with bottom for inactive lanes.
This definition is total at D=0; execution will separately require 0<D. -/
noncomputable def rowMaximum {D : Nat} (values : Fin D → WithBot ℝ) : WithBot ℝ :=
  Finset.univ.sup values

noncomputable def shifted {D : Nat} (values : Fin D → WithBot ℝ)
    (lane : Fin D) : WithBot ℝ :=
  Option.map₂ (fun a b : ℝ => a - b) (values lane) (rowMaximum values)

noncomputable def numerator {D : Nat} (values : Fin D → WithBot ℝ)
    (lane : Fin D) : ℝ :=
  match shifted values lane with
  | none => 0
  | some z => Real.exp z

noncomputable def denominator {D : Nat} (values : Fin D → WithBot ℝ) : ℝ :=
  ∑ j, numerator values j

noncomputable def extendedFormula {D : Nat} (K row : Nat) (log causal : Bool)
    (mode : MaskMode) (xs masks : Fin D → ℝ) (lane : Fin D) : WithBot ℝ :=
  let values := score K row causal mode xs masks
  if log then
    Option.map (fun z => z - Real.log (denominator values)) (shifted values lane)
  else some (numerator values lane / denominator values)

/-- The existing real-memory observer. It intentionally does not assert
that a LOG/CAUSAL future lane has native numerical value zero. -/
noncomputable def formula {D : Nat} (K row : Nat) (log causal : Bool)
    (mode : MaskMode) (xs masks : Fin D → ℝ) (lane : Fin D) : ℝ :=
  WithBot.unbotD 0 (extendedFormula K row log causal mode xs masks lane)

theorem score_congr {D : Nat} (K row : Nat) (causal : Bool)
    (mode : MaskMode) (xs xs' masks masks' : Fin D → ℝ)
    (hx : ∀ j, readActive K row causal j.val = Bool.true → xs j = xs' j)
    (hm : ∀ j, mode.present = Bool.true →
      readActive K row causal j.val = Bool.true → masks j = masks' j) :
    score K row causal mode xs masks = score K row causal mode xs' masks' := by
  funext j
  by_cases hj : readActive K row causal j.val = Bool.true
  · by_cases hp : mode.present = Bool.true
    · simp only [score, hj, hp, if_true, hx j hj, hm j hp hj]
    · simp only [score, hj, hp, Bool.false_eq_true, if_true, if_false, hx j hj]
  · simp only [score, hj, Bool.false_eq_true, if_false]

theorem formula_congr {D : Nat} (K row : Nat) (log causal : Bool)
    (mode : MaskMode) (xs xs' masks masks' : Fin D → ℝ)
    (hx : ∀ j, readActive K row causal j.val = Bool.true → xs j = xs' j)
    (hm : ∀ j, mode.present = Bool.true →
      readActive K row causal j.val = Bool.true → masks j = masks' j)
    (i : Fin D) :
    formula K row log causal mode xs masks i =
      formula K row log causal mode xs' masks' i := by
  simp only [formula, extendedFormula, score_congr K row causal mode xs xs' masks masks' hx hm]

theorem absent_mask_irrelevant {D : Nat} (K row : Nat) (log causal : Bool)
    (xs masks masks' : Fin D → ℝ) (i : Fin D) :
    extendedFormula K row log causal .absent xs masks i =
      extendedFormula K row log causal .absent xs masks' i := by
  simp only [extendedFormula, score, MaskMode.present]
  rfl

theorem causal_log_future {D : Nat} (K row : Nat) (mode : MaskMode)
    (xs masks : Fin D → ℝ) (i : Fin D) (hi : row < i.val) :
    extendedFormula K row Bool.true Bool.true mode xs masks i = none := by
  have hn : ¬i.val ≤ row := Nat.not_le.mpr hi
  simp [extendedFormula, shifted, score, readActive, hn]

theorem causal_plain_future {D : Nat} (K row : Nat) (mode : MaskMode)
    (xs masks : Fin D → ℝ) (i : Fin D) (hi : row < i.val) :
    extendedFormula K row Bool.false Bool.true mode xs masks i = some (0 : ℝ) := by
  have hn : ¬i.val ≤ row := Nat.not_le.mpr hi
  simp [extendedFormula, numerator, shifted, score, readActive, hn]

#axiomsClean mask_conditions
#axiomsClean score_congr
#axiomsClean formula_congr
#axiomsClean absent_mask_irrelevant
#axiomsClean causal_log_future
#axiomsClean causal_plain_future
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward
