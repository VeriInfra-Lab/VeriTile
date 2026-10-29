import bench.tritonbench_g.uniform_sampling.Proofs.Execution
import bench.tritonbench_g.uniform_sampling.Proofs.FlatMemory

/-! Memory safety of the complete load-free prefix, without assuming that
any random or arithmetic statement succeeds. The separate store proof checks
the actual masked output addresses reached after this prefix.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.SafetyPrefix
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000

theorem list_safe_of_each (bounds : RegionBounds) (body : List Stmt)
    (h : ∀ st ∈ body, ∀ s, Stmt.TraceSafe bounds st s) (s : BlockState) :
    Stmt.TraceSafeList bounds body s := by
  induction body generalizing s with
  | nil => simp [Stmt.TraceSafeList]
  | cons head rest ih =>
    rw [Stmt.TraceSafeList]
    refine ⟨h head (by simp) s, ?_⟩
    cases he : stepStmt head s with
    | none => trivial
    | some next => exact ih (fun st hm => h st (by simp [hm])) next

theorem tuple_safe (bounds : RegionBounds) (B : Nat) (s : BlockState) :
    Stmt.TraceSafe bounds (Algorithm.tupleBlock B) s := by
  rw [TupleExecution.tupleBlock_eq]
  simp only [Stmt.TraceSafe, Op.SafeAt, evalOp.eq_def, Tile.scalar, bind, Option.bind,
    if_true, true_and]
  apply list_safe_of_each
  intro st hm t
  simp only [TupleExecution.captureBody, TupleExecution.outputBody, List.mem_append,
    List.mem_cons, List.mem_nil_iff, or_false] at hm
  rcases hm with (rfl | rfl | rfl | rfl | rfl) | (rfl | rfl | rfl | rfl)
  all_goals simp [Stmt.TraceSafe, Op.SafeAt, TupleExecution.fieldStmt, TupleExecution.fieldOp]
  all_goals simp [Op.SafeAt.eq_def]

theorem prefix_safe (bounds : RegionBounds) (swapped : Bool) (y : RegionName) (N : Nat)
    (seed offset : Int) (lower upper : ℝ) (B : Nat) (s : BlockState) :
    Stmt.TraceSafeList bounds
      ((Algorithm.kernel swapped y N seed offset lower upper B).body.take 19) s := by
  apply list_safe_of_each
  intro st hm t
  simp only [Algorithm.kernel, List.take_succ_cons, List.take_zero,
    List.mem_cons, List.mem_nil_iff, or_false] at hm
  rcases hm with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
    rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals first
    | exact tuple_safe bounds B t
    | simp [Stmt.TraceSafe, Op.SafeAt, Algorithm.indexOp, Algorithm.affineStmt,
        Op.intToUnsigned, Op.natToUnsigned, Op.intBitAnd, Op.unsignedToInt, Op.intShiftRight]
  all_goals cases swapped <;> simp [Op.SafeAt]

theorem safe_append (bounds : RegionBounds) (before after : List Stmt) (s : BlockState) :
    Stmt.TraceSafeList bounds (before ++ after) s ↔
      Stmt.TraceSafeList bounds before s ∧
        ∀ t, stepStmts before s = some t → Stmt.TraceSafeList bounds after t := by
  induction before generalizing s with
  | nil => simp [Stmt.TraceSafeList]
  | cons st rest ih =>
    cases he : stepStmt st s <;> simp [Stmt.TraceSafeList, stepStmts, he, ih, and_assoc]

theorem flatten_ok (swapped : Bool) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : ℝ) (B : Nat) :
    (Algorithm.kernel swapped y N seed offset lower upper B).FlattenOk := by
  cases swapped <;> simp [Kernel.FlattenOk, Algorithm.kernel, Algorithm.indexOp, Algorithm.tupleBlock,
    Algorithm.affineStmt, Algorithm.storeStmt, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk,
    Op.intToUnsigned, Op.natToUnsigned, Op.intBitAnd, Op.unsignedToInt, Op.intShiftRight]
  all_goals simp [Op.FlattenOk.eq_def]

#axiomsClean list_safe_of_each
#axiomsClean tuple_safe
#axiomsClean prefix_safe
#axiomsClean safe_append
#axiomsClean flatten_ok

end
end VeriTile.Bench.TritonBenchG.UniformSampling.SafetyPrefix
