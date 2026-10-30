import bench.tritonbench_g.attention_fwd_triton3.IO
import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealImplementation

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.LogicalReader
open VeriTile Triton
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
open Host
noncomputable section

theorem maximum_eq (f : List (ℝ × ℝ) → WithBot ℝ) (empty : f [] = ⊥)
    (cons : ∀ x xs, f (x::xs) = (x.1 : WithBot ℝ) ⊔ f xs) : f = RealFormula.maxScore := by
  funext pairs
  induction pairs with
  | nil => exact empty
  | cons pair pairs ih => rw [cons,RealFormula.maxScore,ih]

theorem statement_iff (c : Call) (f : LogicalIO.Inputs c → LogicalIO.Outputs c)
    (connection : (LogicalIO.io c).lift f = RealInterface.formula c) :
    (LogicalIO.io c).Implements f ↔
      (RealInterface.programIO c).Implements (RealInterface.formula c) := by
  change (RealInterface.programIO c).Implements ((LogicalIO.io c).lift f) ↔ _
  rw [connection]

theorem correct (c : Call) (valid : Valid c) (f : LogicalIO.Inputs c → LogicalIO.Outputs c)
    (connection : (LogicalIO.io c).lift f = RealInterface.formula c) :
    (LogicalIO.io c).Implements f :=
  (statement_iff c f connection).mpr (RealInterface.implements c valid)

/-- All three output channels are connected in mathematical program/row/feature
coordinates before the inverse flat output map is used. -/
theorem formula_connection (c : Call) (f : LogicalIO.Inputs c → LogicalIO.Outputs c)
    (center : ∀ raw program row,
      (f (LogicalIO.inputsEquiv c raw)).center program row =
        (RealContract.expected c.BM c.BN c.D c.finish (c.tile program)
          c.params (RealInterface.localInputs c raw program)).center row)
    (denominator : ∀ raw program row,
      (f (LogicalIO.inputsEquiv c raw)).denominator program row =
        (RealContract.expected c.BM c.BN c.D c.finish (c.tile program)
          c.params (RealInterface.localInputs c raw program)).denominator row)
    (output : ∀ raw program row col,
      (f (LogicalIO.inputsEquiv c raw)).output program row col =
        (RealContract.expected c.BM c.BN c.D c.finish (c.tile program)
          c.params (RealInterface.localInputs c raw program)).output (row,col,())) :
    (LogicalIO.io c).lift f = RealInterface.formula c := by
  funext raw o j
  fin_cases o
  · exact center raw _ _
  · exact denominator raw _ _
  · exact output raw _ _ _

/-- The independently presented recursive maximum is fixed by its empty/cons
equations. The checked pointwise connections cover all three full channels. -/
theorem correct_with_maximum (c : Call) (valid : Valid c)
    (maximum : List (ℝ × ℝ) → WithBot ℝ) (empty : maximum [] = ⊥)
    (cons : ∀ x xs, maximum (x::xs) = (x.1 : WithBot ℝ) ⊔ maximum xs)
    (f : (List (ℝ × ℝ) → WithBot ℝ) → LogicalIO.Inputs c → LogicalIO.Outputs c)
    (center : ∀ raw program row,
      (f RealFormula.maxScore (LogicalIO.inputsEquiv c raw)).center program row =
        (RealContract.expected c.BM c.BN c.D c.finish (c.tile program)
          c.params (RealInterface.localInputs c raw program)).center row)
    (denominator : ∀ raw program row,
      (f RealFormula.maxScore (LogicalIO.inputsEquiv c raw)).denominator program row =
        (RealContract.expected c.BM c.BN c.D c.finish (c.tile program)
          c.params (RealInterface.localInputs c raw program)).denominator row)
    (output : ∀ raw program row col,
      (f RealFormula.maxScore (LogicalIO.inputsEquiv c raw)).output program row col =
        (RealContract.expected c.BM c.BN c.D c.finish (c.tile program)
          c.params (RealInterface.localInputs c raw program)).output (row,col,())) :
    (LogicalIO.io c).Implements (f maximum) := by
  rw [maximum_eq maximum empty cons]
  exact correct c valid _ (formula_connection c _ center denominator output)

#axiomsClean maximum_eq
#axiomsClean statement_iff
#axiomsClean formula_connection
#axiomsClean correct
#axiomsClean correct_with_maximum
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.LogicalReader
