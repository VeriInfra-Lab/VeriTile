import bench.tritonbench_g.ksoftmax_triton.Proofs.Mathematical_BackwardSource
import VeriTile.Triton.Float.ExpressionRewrite
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.Floating
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

@[spec_rule] theorem index_product_commutes {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before : List ComputeStmt) (a b : Op .nat []) :
    OpEquivalent M s before (.mul .nat .nil a b) (.mul .nat .nil b a) := by
  intro t _
  change (evalOp (engine M) none a t >>= fun x =>
    evalOp (engine M) none b t >>= fun y => some (bop Nat.mul .nil x y)) =
    (evalOp (engine M) none b t >>= fun y =>
    evalOp (engine M) none a t >>= fun x => some (bop Nat.mul .nil y x))
  cases ha : evalOp (engine M) none a t <;>
    cases hb : evalOp (engine M) none b t
  all_goals first
    | rfl
    | apply congrArg some
      funext i
      exact Nat.mul_comm _ _

@[spec_rule] theorem pointer_add_congr {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before : List ComputeStmt) (bc : Broadcast sa sb sh)
    (p p' : Op .ptr sa) (a a' : Op .nat sb)
    (hp : OpEquivalent M s before p p') (ha : OpEquivalent M s before a a') :
    OpEquivalent M s before (.ptrAdd bc p a) (.ptrAdd bc p' a') := by
  intro t ht
  simp only [FP.Structural.evalOp]
  rw [hp t ht, ha t ht]

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def program (storage : Typed.Storage) (commuted : Bool) (GradIn GradOut Out : RegionName)
    (bm bn gm gn om on K D : Nat) (log causal fp16 : Bool) : Program where
  kernel := Typed.homogeneousKernel storage commuted GradIn GradOut Out
    bm bn gm gn om on K D log causal fp16
  domain := []

/-- The complete typed programs have identical execution for every abstract
FP algebra, including failure and the entire final state. Only integer
address multiplication is commuted; no floating operation is reassociated.
Successful executions are a separate obligation, not implied by this equality. -/
specification backward_index_equiv (storage : Typed.Storage) (GradIn GradOut Out : RegionName)
    (bm bn gm gn om on K D : Nat) (log causal fp16 : Bool) :
    program storage Bool.false GradIn GradOut Out bm bn gm gn om on K D log causal fp16
      ≡[assumptions]
    program storage Bool.true GradIn GradOut Out bm bn gm gn om on K D log causal fp16 := by
  cases storage <;> equiv_decompose <;> fp_prove

#print_fp_assumptions backward_index_equiv
#axiomsClean index_product_commutes
#axiomsClean pointer_add_congr
#axiomsClean backward_index_equiv
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.Floating
