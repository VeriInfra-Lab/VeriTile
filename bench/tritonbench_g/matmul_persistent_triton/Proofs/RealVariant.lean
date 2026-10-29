import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealVariantSource
import VeriTile.Triton.Memory.TraceRewrite

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealVariant
open VeriTile.Triton RealVariantSource
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem row_eval (p : Parameters) (v : Bool) (s : BlockState) :
    evalOp (rowOp p v) s = evalOp (rowOp p Bool.false) s := by
  cases v
  · rfl
  · simp [rowOp,evalOp.eq_def,bind,Option.bind,Tile.scalar,Tile.bop,
      NumericDType.mul,NumericDType.add,Broadcast.leftIndex,Broadcast.rightIndex,Int.mul_comm,-Tile.scalar_eta]

theorem row_exec (p : Parameters) (v : Bool) (s : BlockState) :
    stepStmt (rowStatement p v) s = stepStmt (rowStatement p Bool.false) s := by
  simp only [rowStatement,stepStmt,row_eval]

theorem row_safe (p : Parameters) (v : Bool) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafe bounds (rowStatement p v) s := by
  cases v <;> simp [rowStatement,rowOp,Stmt.TraceSafe,Op.SafeAt.eq_def]

theorem store_exec (p : Parameters) (v : Bool) (s : BlockState) :
    stepStmts (store p v) s = stepStmts (store p Bool.false) s := by
  simp only [store,stepStmts,row_exec]

theorem store_safe (p : Parameters) (v : Bool) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (store p v) s ↔
      Stmt.TraceSafeList bounds (store p Bool.false) s := by
  simp only [store,Stmt.TraceSafeList,row_safe,row_exec]

theorem finish_exec (p : Parameters) (v : Bool) (s : BlockState) :
    stepStmt (.ifThen RealFinish.condition (store p v)) s =
      stepStmt (.ifThen RealFinish.condition (store p Bool.false)) s := by
  simp only [stepStmt,store_exec]

theorem finish_safe (p : Parameters) (v : Bool) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafe bounds (.ifThen RealFinish.condition (store p v)) s ↔
      Stmt.TraceSafe bounds (.ifThen RealFinish.condition (store p Bool.false)) s := by
  simp only [Stmt.TraceSafe,store_safe]

theorem body_exec (p : Parameters) (v : Bool) (s : BlockState) :
    stepStmts (body p v) s = stepStmts (body p Bool.false) s := by
  simp only [body,ProgramRewrite.append,stepStmts,finish_exec]

theorem body_safe (p : Parameters) (v : Bool) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (body p v) s ↔
      Stmt.TraceSafeList bounds (body p Bool.false) s := by
  simp only [body,TraceRewrite.append_iff,Stmt.TraceSafeList,finish_safe,finish_exec]

theorem loop_exec (p : Parameters) (v : Bool) (s : BlockState) :
    stepStmt (loop p v) s = stepStmt (loop p Bool.false) s :=
  ProgramRewrite.dynamic_range_congr _ _ _ _ _ _ (body_exec p v) s

theorem loop_safe (p : Parameters) (v : Bool) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafe bounds (loop p v) s ↔ Stmt.TraceSafe bounds (loop p Bool.false) s :=
  TraceRewrite.dynamic_range_congr bounds _ _ _ _ _ _ (body_exec p v) (body_safe p v bounds) s

/-- Equality of complete real runs, including arbitrary initial registers and
failure results. Only signed integer address arithmetic is commuted. -/
theorem equivalent (p : Parameters) (v : Bool) (s : BlockState) :
    exec (RealSource.kernel p v) s = exec (RealSource.kernel p Bool.false) s := by
  simp only [exec,matches_source,ProgramRewrite.append,stepStmts,loop_exec]

/-- Safety is transported along the actual reached states; final-state
execution equality alone is not used as an access-safety argument. -/
theorem safe_iff (p : Parameters) (v : Bool) (bounds : RegionBounds) (s : BlockState) :
    (RealSource.kernel p v).TraceSafe bounds s ↔
      (RealSource.kernel p Bool.false).TraceSafe bounds s := by
  simp only [Kernel.TraceSafe,matches_source,TraceRewrite.append_iff,
    Stmt.TraceSafeList,loop_safe,loop_exec]

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealVariant
