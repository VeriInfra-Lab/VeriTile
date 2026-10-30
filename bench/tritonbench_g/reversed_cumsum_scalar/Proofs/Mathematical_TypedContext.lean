import VeriTile.Triton.Float.Loop
import VeriTile.Meta.StatementAudit

/-! Context lemmas for the candidate's sole change: commuting the natural
row-stride product in the input block pointer. All floating expressions and
all state components stay in their original evaluation context. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.TypedContext
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1800000
set_option maxRecDepth 16000

def pointerStmt (swapped : Bool) (r : RegionName) (T BT : Nat)
    (offset : Op .nat []) : ComputeStmt :=
  .assign .blockPtr [BT] "p_s" (.alg (.makeBlockPtrDynOffsets r
    (if swapped then .mul .nat .nil (.constNat T) (.ref .nat [] "i_bh")
      else .mul .nat .nil (.ref .nat [] "i_bh") (.constNat T))
    [T] [BT] [1] [offset]))

theorem pointer_step (r : RegionName) (T BT : Nat) (offset : Op .nat [])
    {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    step M (pointerStmt Bool.false r T BT offset) s =
      step M (pointerStmt Bool.true r T BT offset) s := by
  cases h : s.regs .nat [] "i_bh" <;>
    simp [pointerStmt, step, FP.Structural.evalExpr, evalOp_unfold, h, numeric,
      bop, Nat.mul_comm]

theorem run_insert {α : Type} [Inhabited α] (M : Algebra α)
    (before after : List ComputeStmt) (a b : ComputeStmt)
    (h : ∀ s, step M a s = step M b s) (s : State α) :
    run M (before ++ a :: after) s = run M (before ++ b :: after) s := by
  rw [run_append, run_append]
  cases hp : run M before s <;>
    simp only [Option.bind_none, Option.bind_some, run, h]

theorem signed_loop {α : Type} [Inhabited α] (M : Algebra α)
    (idx : RegName) (start stop : Op .int []) (stride : Op .nat [])
    (lhs rhs : List ComputeStmt) (hbody : ∀ s, run M lhs s = run M rhs s)
    (s : State α) :
    step M (.forRangeInt idx start stop stride lhs) s =
      step M (.forRangeInt idx start stop stride rhs) s := by
  simp only [step]
  cases ha : FP.Structural.evalOp M none start s with
  | none => rfl
  | some a =>
    cases hb : FP.Structural.evalOp M none stop s with
    | none => rfl
    | some b =>
      cases hd : FP.Structural.evalOp M none stride s with
      | none => rfl
      | some d =>
        simp only [Option.bind_some, Option.bind_eq_bind]
        apply rangeInt_congr_on_reachable
        intro cur t _ _ _
        exact hbody _

#axiomsClean pointer_step
#axiomsClean run_insert
#axiomsClean signed_loop
end
end VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.TypedContext
