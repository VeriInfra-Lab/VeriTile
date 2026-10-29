import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardTileBridge

/-! Draft scalar bridge for the actual sequence of masked X load, causal
where, optional masked M load and addition. This does not replace the source
program or provide an execution witness; it is the arithmetic identity needed
when observing the already reduced source execution.
-/
namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward
open VeriTile.Triton
noncomputable section

def loadedScore {D : Nat} (K row : Nat) (causal : Bool) (mode : MaskMode)
    (xs masks : Fin D → ℝ) (i : Fin D) : WithBot ℝ :=
  let x : WithBot ℝ := if readActive K row causal i.val then some (xs i) else none
  let x := if causal then (if row < i.val then none else x) else x
  if mode.present then
    WithBot.realAdd x (if readActive K row causal i.val then some (masks i) else none)
  else x

/-- The redundant source causal where is preserved in loadedScore, then
proved to leave precisely the original read mask's score. -/
theorem loadedScore_eq_score {D : Nat} (K row : Nat) (causal : Bool)
    (mode : MaskMode) (xs masks : Fin D → ℝ) :
    loadedScore K row causal mode xs masks = score K row causal mode xs masks := by
  funext i
  cases mode <;> cases causal
  all_goals
    by_cases hK : i.val < K
    · by_cases hrow : i.val ≤ row
      · simp [loadedScore, score, readActive, MaskMode.present,
          hK, hrow, Nat.not_lt.mpr hrow, WithBot.realAdd]
      · simp [loadedScore, score, readActive, MaskMode.present,
          hK, hrow, Nat.lt_of_not_ge hrow, WithBot.realAdd]
    · by_cases hrow : i.val ≤ row
      · simp [loadedScore, score, readActive, MaskMode.present,
          hK, hrow, Nat.not_lt.mpr hrow, WithBot.realAdd]
      · simp [loadedScore, score, readActive, MaskMode.present,
          hK, hrow, Nat.lt_of_not_ge hrow, WithBot.realAdd]

theorem loadedScore_result_formula (K row D : Nat) (log causal : Bool)
    (mode : MaskMode) (xs masks : Fin D → ℝ) (i : Fin D) :
    (resultTile log (loadedScore K row causal mode xs masks)).data (i, PUnit.unit) =
      extendedFormula K row log causal mode xs masks i := by
  rw [loadedScore_eq_score]
  exact resultTile_extendedFormula K row D log causal mode xs masks i

end
#axiomsClean loadedScore_eq_score
#axiomsClean loadedScore_result_formula
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward
