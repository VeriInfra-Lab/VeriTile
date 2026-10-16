import bench.tritonbench_g.kldiv_ops.Proofs.SourceProjection

namespace VeriTile.Bench.TritonBenchG.KldivOps.Source
open VeriTile.Triton TypedSource
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

theorem forward_math_original (Y T O : RegionName) (sy st so N B : Nat)
    (eps : ℝ) (flag : Bool) (reduction : Nat) :
    forwardMath Bool.false Y T O sy st so N B eps flag reduction =
      FullAlgorithm.forward Y T O sy st so N B eps flag reduction := by
  cases flag <;> simp only [forwardMath, forwardPrefix, rowPointer,
    FullAlgorithm.forward, FullAlgorithm.forwardSurface,
    ComputeKernel.eraseDType, ComputeKernel.fromKernelBody,
    ComputeKernel.toAlgKernel_fromKernelBody, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType, Bool.false_eq_true, ↓reduceIte,
    List.drop, List.cons_append, List.nil_append]
  all_goals rfl

theorem backward_math_original (T O : RegionName) (st so N B : Nat) (flag : Bool) :
    backwardMath Bool.false T O st so N B flag =
      FullAlgorithm.backward T O st so N B flag := by
  cases flag <;> rfl

theorem forward_prefix_flat (A : FlatAlloc) (Y T O : RegionName)
    (sy st so : Nat) (s : BlockState) :
    stepStmts (A.flattenStmts (forwardPrefix Bool.true Y T O sy st so)) s =
      stepStmts (A.flattenStmts (forwardPrefix Bool.false Y T O sy st so)) s := by
  simp [forwardPrefix, rowPointer, FlatAlloc.flattenStmts, FlatAlloc.flattenStmt,
    FlatAlloc.flattenOp.eq_def, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.uop, NumericDType.mul, Int.mul_comm]

theorem backward_prefix_flat (A : FlatAlloc) (T O : RegionName)
    (st so : Nat) (s : BlockState) :
    stepStmts (A.flattenStmts (backwardPrefix Bool.true T O st so)) s =
      stepStmts (A.flattenStmts (backwardPrefix Bool.false T O st so)) s := by
  simp [backwardPrefix, rowPointer, FlatAlloc.flattenStmts, FlatAlloc.flattenStmt,
    FlatAlloc.flattenOp.eq_def, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.uop, NumericDType.mul, Int.mul_comm]

theorem forward_math_flat_exec (A : FlatAlloc) (v : Bool) (Y T O : RegionName)
    (sy st so N B : Nat) (eps : ℝ) (flag : Bool) (reduction : Nat) (s : BlockState) :
    exec (A.flattenKernel (forwardMath v Y T O sy st so N B eps flag reduction)) s =
      exec (A.flattenKernel (FullAlgorithm.forward Y T O sy st so N B eps flag reduction)) s := by
  rw [← forward_math_original Y T O sy st so N B eps flag reduction]
  cases v
  · rfl
  · simp only [exec, FlatAlloc.flattenKernel, forwardMath, FlatAlloc.flattenStmts_append,
      ProgramRewrite.append, forward_prefix_flat]

theorem backward_math_flat_exec (A : FlatAlloc) (v : Bool) (T O : RegionName)
    (st so N B : Nat) (flag : Bool) (s : BlockState) :
    exec (A.flattenKernel (backwardMath v T O st so N B flag)) s =
      exec (A.flattenKernel (FullAlgorithm.backward T O st so N B flag)) s := by
  rw [← backward_math_original T O st so N B flag]
  cases v
  · rfl
  · simp only [exec, FlatAlloc.flattenKernel, backwardMath, FlatAlloc.flattenStmts_append,
      ProgramRewrite.append, backward_prefix_flat]

theorem forward_default_flat_exec (A : FlatAlloc) (pred target output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat) (s : BlockState) :
    exec (A.flattenKernel (forwardDefault pred target output v Y T O sy st so N B eps reduction).eraseDType.toAlgKernel) s =
      exec (A.flattenKernel (FullAlgorithm.forward Y T O sy st so N B eps Bool.false reduction)) s := by
  rw [forward_default_projection]
  exact forward_math_flat_exec A v Y T O sy st so N B eps Bool.false reduction s

theorem forward_log_flat_exec (A : FlatAlloc) (pred : Storage) (target : MathStorage) (output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat) (s : BlockState) :
    exec (A.flattenKernel (forwardLog pred target output v Y T O sy st so N B eps reduction).eraseDType.toAlgKernel) s =
      exec (A.flattenKernel (FullAlgorithm.forward Y T O sy st so N B eps Bool.true reduction)) s := by
  rw [forward_log_projection]
  exact forward_math_flat_exec A v Y T O sy st so N B eps Bool.true reduction s

theorem backward_default_flat_exec (A : FlatAlloc) (target output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) (s : BlockState) :
    exec (A.flattenKernel (backwardDefault target output v T O st so N B).eraseDType.toAlgKernel) s =
      exec (A.flattenKernel (FullAlgorithm.backward T O st so N B Bool.false)) s := by
  rw [backward_default_projection]
  exact backward_math_flat_exec A v T O st so N B Bool.false s

theorem backward_log_flat_exec (A : FlatAlloc) (target : MathStorage) (output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) (s : BlockState) :
    exec (A.flattenKernel (backwardLog target output v T O st so N B).eraseDType.toAlgKernel) s =
      exec (A.flattenKernel (FullAlgorithm.backward T O st so N B Bool.true)) s := by
  rw [backward_log_projection]
  exact backward_math_flat_exec A v T O st so N B Bool.true s

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KldivOps.Source
