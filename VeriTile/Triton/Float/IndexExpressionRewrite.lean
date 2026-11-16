import VeriTile.Triton.Float.ExpressionRewrite

/-! Exact structural rewrites for natural-number address expressions. -/

namespace VeriTile.Triton.FP.ProfiledRewrite
open Structural

/-- Natural index multiplication commutes through evaluation, including failed operands. -/
theorem op_nat_mul_comm {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before : List ComputeStmt) (a b : Op .nat []) :
    OpEquivalent M s before (.mul .nat .nil a b) (.mul .nat .nil b a) := by
  intro t _
  change (Structural.evalOp (engine M) none a t >>= fun x =>
    Structural.evalOp (engine M) none b t >>= fun y => some (bop Nat.mul .nil x y)) =
    (Structural.evalOp (engine M) none b t >>= fun y =>
    Structural.evalOp (engine M) none a t >>= fun x => some (bop Nat.mul .nil y x))
  cases ha : Structural.evalOp (engine M) none a t <;>
    cases hb : Structural.evalOp (engine M) none b t
  all_goals first
    | rfl
    | apply congrArg some
      funext i
      exact Nat.mul_comm _ _

/-- Equal pointer and index expressions preserve address addition. -/
theorem op_ptr_add_congr {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before : List ComputeStmt) (bc : Broadcast sa sb sh)
    (p p' : Op .ptr sa) (a a' : Op .nat sb)
    (hp : OpEquivalent M s before p p') (ha : OpEquivalent M s before a a') :
    OpEquivalent M s before (.ptrAdd bc p a) (.ptrAdd bc p' a') := by
  intro t ht
  simp only [FP.Structural.evalOp]
  rw [hp t ht, ha t ht]

end VeriTile.Triton.FP.ProfiledRewrite
