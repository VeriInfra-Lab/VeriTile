import VeriTile.Triton.Float.EvalOpR
import VeriTile.Meta.StatementAudit

/-! The mathematical projection of the true floating absolute-value node is
compatible with the historical Real comparison/subtraction expression.
This statement concerns `WithBot ℝ`, not signed-zero or NaN bit behavior. -/
namespace VeriTile.Triton

private theorem absWithBot_projection (x : WithBot ℝ) :
    x.map (fun r => |r|) =
      if ComparableDType.lt .real x (some 0) then WithBot.realSub (some 0) x else x := by
  cases x with
  | bot => rfl
  | coe r =>
    by_cases h : r < 0
    · simp only [ComparableDType.lt, WithBot.some_eq_coe, WithBot.coe_lt_coe, h, decide_true, Bool.true_eq, ↓reduceIte, WithBot.map_coe, abs_of_neg h]
      change (some (-r) : WithBot ℝ) = some (0 - r)
      simp
    · simp [ComparableDType.lt, WithBot.some_eq_coe, h, abs_of_nonneg (le_of_not_gt h), WithBot.realSub]

/-- The new abs node preserves the old mathematical meaning on every nonempty shape
and state, including bottom and failed operand evaluations. -/
theorem evalOp_abs_real_projection {n : Nat} {tail : TileShape} (a : Op .real (n :: tail)) (s : BlockState) :
    evalOp (.absFloat .real a) s =
      evalOp (.where (.lt .real .scalarR a (.const 0))
        (.sub .real .scalarL (.const 0) a) a) s := by
  conv_lhs => rw [evalOp.eq_def]
  cases h : evalOp a s with
  | none => simp [evalOp, FloatDType.toTileDType, h, Option.bind]
  | some v =>
    simp only [evalOp, FloatDType.toTileDType, h, Option.bind_none, Option.bind_some, bind, Option.bind]
    congr 1
    congr 1
    funext i
    exact absWithBot_projection (v.data i)

#axiomsClean evalOp_abs_real_projection
end VeriTile.Triton
