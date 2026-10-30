import bench.tritonbench_g.attn_fwd_triton.Proofs.FlatCongruence
import bench.tritonbench_g.attn_fwd_triton.Proofs.AttnFwdTriton_Verified_Execution_Execution
import bench.tritonbench_g.attn_fwd_triton.Proofs.Relocation

/-! Exact full-source decomposition after pointer-base relocation. Only the
six pointer constructors change; the actual arithmetic and loops are retained. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.FlatSource
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttnFwdCausal
set_option maxHeartbeats 1500000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem append (A : FlatAlloc) (a b : List Stmt) :
    A.flattenStmts (a++b) = A.flattenStmts a ++ A.flattenStmts b := by
  induction a with
  | nil => rfl
  | cons st a ih => simp only [List.cons_append,FlatAlloc.flattenStmts,ih]

theorem coordinates (A : FlatAlloc) (p : Parameters) (BM BN : Nat) :
    A.flattenStmts (Coordinates.code p BM BN) = Coordinates.code p BM BN := by
  simp only [Coordinates.code,Coordinates.ceilExpr,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,FlatCongruence.vec0,FlatCongruence.vec1,FlatCongruence.dot2,FlatCongruence.sum1,FlatCongruence.max1]

theorem initialization (A : FlatAlloc) (BM BD D : Nat) :
    A.flattenStmts (Initialization.code BM BD D) = Initialization.code BM BD D := by
  simp [Initialization.code,RealInit.code,queryMask,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,
    FlatAlloc.flattenOp,FlatCongruence.vec0,FlatCongruence.vec1,FlatCongruence.dot2,FlatCongruence.sum1,FlatCongruence.max1,FlatAlloc.flattenAccess,FlatAlloc.flattenMask]
  all_goals repeat' apply And.intro
  all_goals erw [FlatAlloc.flattenOp]; simp only [FlatAlloc.flattenOp]

theorem loops (A : FlatAlloc) (BM BN BD D : Nat) (diagonal : Bool) :
    A.flattenStmts (TwoLoops.code BM BN BD D diagonal) = TwoLoops.code BM BN BD D diagonal := by
  cases diagonal <;> simp [TwoLoops.code,FirstSetup.code,FirstSetup.copies,RealFirstSetup.copies,
    Setup.code,Setup.lowerExpr,Setup.upperExpr,RealSetup.lowerExpr,RealSetup.upperExpr,
    loop,sourceIteration,flatIteration,loadPrefix,keyMask,valueMask,tail,guard,
    RealBody.score,RealBody.scoreExpr,RealBody.center,RealBody.centerExpr,RealBody.probability,
    suffix,RealBody.rescale,valueLoad,RealBody.accumulate,advance,RealExpressions.maskExpr,
    FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,FlatCongruence.vec0,FlatCongruence.vec1,FlatCongruence.dot2,FlatCongruence.sum1,FlatCongruence.max1,FlatAlloc.flattenAccess,FlatAlloc.flattenMask]
  all_goals repeat' apply And.intro
  all_goals erw [FlatAlloc.flattenOp]; simp only [FlatAlloc.flattenOp]

theorem epilogue (A : FlatAlloc) (BM BD D : Nat) (reciprocal : Bool) :
    A.flattenStmts (Verified.epilogue BM BD D reciprocal) = Verified.epilogue BM BD D reciprocal := by
  cases reciprocal <;> simp [Verified.epilogue,RealSource.normalization,queryMask,
    FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,FlatCongruence.vec0,FlatCongruence.vec1,FlatCongruence.dot2,FlatCongruence.sum1,FlatCongruence.max1,FlatAlloc.flattenAccess,FlatAlloc.flattenMask]
  all_goals repeat' apply And.intro
  all_goals erw [FlatAlloc.flattenOp]; simp only [FlatAlloc.flattenOp]

theorem body (A : FlatAlloc) (p : Parameters) (BM BN BD : Nat) (reciprocal : Bool) :
    A.flattenStmts (Verified.code BM BN BD p reciprocal) =
      (Coordinates.code p BM BN ++ A.flattenStmts (Pointers.code p BM BN BD) ++ Initialization.code BM BD p.headDim) ++
      TwoLoops.code BM BN BD p.headDim (decide (p.stage=2)) ++ Verified.epilogue BM BD p.headDim reciprocal := by
  rw [Execution.body,Coordinates.source,Pointers.source,Initialization.source]
  simp only [append,coordinates,initialization,loops,epilogue]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.FlatSource
