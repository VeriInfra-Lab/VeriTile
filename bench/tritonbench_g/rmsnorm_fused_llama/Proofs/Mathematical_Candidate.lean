import bench.tritonbench_g.rmsnorm_fused_llama.Proofs.Mathematical_Source
import VeriTile.Triton.Semantics.ProgramRewrite

namespace VeriTile.Bench.TritonBenchG.RmsnormFusedLlama
open VeriTile.Triton
set_option maxHeartbeats 2000000

def pointerPrefix (v : Bool) (Y : RegionName) (stride : Nat) (eps : ℝ) : List Stmt :=
  [.assign .real [] "eps_arg" (.const eps),
   .assign .nat [] "row" (.programId 0),
   .assign .ptr [] "Y" (.ptrAdd .nil (.ptrBase Y)
     (if v then .mul .nat .nil (.constNat stride) (.ref .nat [] "row")
      else .mul .nat .nil (.ref .nat [] "row") (.constNat stride)))]

theorem prefix_source (v : Bool) (X Y W : RegionName) (stride N B : Nat) (eps : ℝ) :
    (mathKernel v X Y W stride N B eps).body.take 3 = pointerPrefix v Y stride eps := by
  cases v <;> simp [mathKernel, typedKernel,
    VeriTile.Bench.Repairs.RmsnormFusedLlama.runtimeKernel,
    VeriTile.Bench.Repairs.RmsnormFusedLlama.runtimeCandidate,
    ComputeKernel.withScalarArguments, ScalarArgument.statement, ScalarArgument.fp32,
    ComputeKernel.eraseDType, ComputeKernel.surfaceBody, ComputeKernel.toAlgKernel,
    ComputeKernel.toAlgorithm?, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType, Except.bind, bind, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    eraseDType, Region.cast,pointerPrefix]

theorem tail_source (X Y W : RegionName) (stride N B : Nat) (eps : ℝ) :
    (mathKernel Bool.true X Y W stride N B eps).body.drop 3 =
      (mathKernel Bool.false X Y W stride N B eps).body.drop 3 := by
  simp [mathKernel, typedKernel,
    VeriTile.Bench.Repairs.RmsnormFusedLlama.runtimeKernel,
    VeriTile.Bench.Repairs.RmsnormFusedLlama.runtimeCandidate,
    ComputeKernel.withScalarArguments, ScalarArgument.statement, ScalarArgument.fp32,
    ComputeKernel.eraseDType, ComputeKernel.surfaceBody, ComputeKernel.toAlgKernel,
    ComputeKernel.toAlgorithm?, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType, Except.bind, bind, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    eraseDType, Region.cast]

theorem prefix_flat (A : FlatAlloc) (Y : RegionName) (stride : Nat) (eps : ℝ) (s : BlockState) :
    stepStmts (A.flattenStmts (pointerPrefix Bool.true Y stride eps)) s =
      stepStmts (A.flattenStmts (pointerPrefix Bool.false Y stride eps)) s := by
  simp [pointerPrefix, FlatAlloc.flattenStmts, FlatAlloc.flattenStmt, FlatAlloc.flattenOp.eq_def,
    stepStmts, stepStmt, evalOp.eq_def, Tile.bop, NumericDType.mul, Nat.mul_comm]

/-- Full state transport changes one address product and retains both loops. -/
theorem typed_flat_exec (A : FlatAlloc) (v : Bool)
    (X Y W : RegionName) (stride N B : Nat) (eps : ℝ) (s : BlockState) :
    exec (A.flattenKernel
      (typedKernel v X Y W stride N B eps).eraseDType.toAlgKernel) s =
      exec (A.flattenKernel (mathSource X Y W stride N B eps).toAlgKernel) s := by
  rw [typed_projection]
  cases v
  · rfl
  · change exec (A.flattenKernel (mathKernel Bool.true X Y W stride N B eps)) s =
      exec (A.flattenKernel (mathKernel Bool.false X Y W stride N B eps)) s
    unfold exec FlatAlloc.flattenKernel
    rw [← List.take_append_drop 3 (mathKernel Bool.true X Y W stride N B eps).body,
      ← List.take_append_drop 3 (mathKernel Bool.false X Y W stride N B eps).body]
    have ht : (mathKernel Bool.true X Y W stride N B eps).body.drop 3 =
        (mathKernel Bool.false X Y W stride N B eps).body.drop 3 := tail_source X Y W stride N B eps
    rw [prefix_source, prefix_source]
    simp only [FlatAlloc.flattenStmts_append, ProgramRewrite.append, prefix_flat, ht]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsnormFusedLlama
