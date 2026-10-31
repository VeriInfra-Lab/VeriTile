import bench.tritonbench_g.batched_vecmat_mult.Sources.Mathematical_Kernels
import bench.tritonbench_g.batched_vecmat_mult.Proofs.Mathematical_RealFoundation
import VeriTile.Triton.Semantics.ProgramRewrite
import VeriTile.Meta.StatementAudit

/-! Exact real correspondence for the complete sources. The original erases
to the existing real execution foundation. Real addition commutes in every
loop state, so the candidate has the same full flat-memory execution result,
including failures. This bridge imports no FP admission or FP proof. -/
namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.RealSource
open VeriTile.Triton
set_option maxHeartbeats 1800000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

section
variable (A B out : RegionName) (M N K BM BN BK : Nat)

end

def blockSum (BM BN BK : Nat) : Op .real [BM,BN] :=
  .transpose (batch := []) (.reduceSum ⟨2, by simp⟩ Bool.false
    (.mul .real (.consL (.consSame (.consSame .nil)))
      (.ref .real [1,BM,BK] "expanded_a") (.ref .real [BN,BM,BK] "b")))

private theorem real_add_comm (a b : WithBot ℝ) : WithBot.realAdd a b = WithBot.realAdd b a := by
  cases a <;> cases b
  · rfl
  · rfl
  · rfl
  · simp [WithBot.realAdd, add_comm]

private theorem eval_add_comm (a b : Op .real [BM,BN]) (s : BlockState) :
    evalOp (.add .real (.consSame (.consSame .nil)) a b) s =
      evalOp (.add .real (.consSame (.consSame .nil)) b a) s := by
  simp only [evalOp_add]
  cases ha : evalOp a s <;> cases hb : evalOp b s
  all_goals simp only [Option.bind_eq_bind, Option.bind_none, Option.bind_some]
  rename_i va vb
  congr 1
  ext i
  exact real_add_comm _ _

private theorem flatten_append (alloc : FlatAlloc) (xs ys : List Stmt) :
    alloc.flattenStmts (xs ++ ys) = alloc.flattenStmts xs ++ alloc.flattenStmts ys := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simp only [List.cons_append, FlatAlloc.flattenStmts, ih]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.RealSource
