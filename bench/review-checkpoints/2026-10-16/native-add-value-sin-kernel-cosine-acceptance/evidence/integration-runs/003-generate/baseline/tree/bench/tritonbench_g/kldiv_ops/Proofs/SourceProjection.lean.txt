import bench.tritonbench_g.kldiv_ops.Proofs.SourceBridgeProjects
import bench.tritonbench_g.kldiv_ops.Proofs.SourceBridgePrefix
import bench.tritonbench_g.kldiv_ops.Proofs.Algorithm

/-! Ideal-real projection of every typed source profile. Erasure is explicit;
no typed cast or rounding node is discarded on the FP equivalence surface. -/
namespace VeriTile.Bench.TritonBenchG.KldivOps.Source
open VeriTile.Triton TypedSource
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def forwardMath (v : Bool) (Y T O : RegionName) (sy st so N B : Nat)
    (eps : ℝ) (flag : Bool) (reduction : Nat) : Kernel :=
  { FullAlgorithm.forward Y T O sy st so N B eps flag reduction with
    body := forwardPrefix v Y T O sy st so ++
      (FullAlgorithm.forward Y T O sy st so N B eps flag reduction).body.drop 4 }

def backwardMath (v : Bool) (T O : RegionName) (st so N B : Nat) (flag : Bool) : Kernel :=
  { FullAlgorithm.backward T O st so N B flag with
    body := backwardPrefix v T O st so ++
      (FullAlgorithm.backward T O st so N B flag).body.drop 3 }

/-- All storage annotations are erased on the real specification surface only. -/
theorem forward_default_projection (pred target output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat) :
    (forwardDefault pred target output v Y T O sy st so N B eps reduction).eraseDType.toAlgKernel =
      forwardMath v Y T O sy st so N B eps Bool.false reduction := by
  simp only [ComputeKernel.eraseDType, forward_default_projects,
    ComputeKernel.toAlgKernel_fromKernelBody]
  cases pred <;> cases target <;> cases output <;> cases v
  all_goals simp only [forwardDefault, forwardDefaultOriginal, commuteOutput, forwardMath, rowPointer, forwardPrefix, FullAlgorithm.forward, FullAlgorithm.forwardSurface, ComputeKernel.eraseDType, ComputeKernel.fromKernelBody, forwardDefault_fp32_fp32, forwardDefault_fp32_fp64, forwardDefault_fp32_fp16, forwardDefault_fp32_bf16, forwardDefault_fp32_f8e4, forwardDefault_fp32_f8e5, forwardDefault_fp64_fp32, forwardDefault_fp64_fp64, forwardDefault_fp64_fp16, forwardDefault_fp64_bf16, forwardDefault_fp64_f8e4, forwardDefault_fp64_f8e5, forwardDefault_fp16_fp32, forwardDefault_fp16_fp64, forwardDefault_fp16_fp16, forwardDefault_fp16_bf16, forwardDefault_fp16_f8e4, forwardDefault_fp16_f8e5, forwardDefault_bf16_fp32, forwardDefault_bf16_fp64, forwardDefault_bf16_fp16, forwardDefault_bf16_bf16, forwardDefault_bf16_f8e4, forwardDefault_bf16_f8e5, forwardDefault_f8e4_fp32, forwardDefault_f8e4_fp64, forwardDefault_f8e4_fp16, forwardDefault_f8e4_bf16, forwardDefault_f8e4_f8e4, forwardDefault_f8e4_f8e5, forwardDefault_f8e5_fp32, forwardDefault_f8e5_fp64, forwardDefault_f8e5_fp16, forwardDefault_f8e5_bf16, forwardDefault_f8e5_f8e4, forwardDefault_f8e5_f8e5,
    List.take, List.drop, List.cons_append, List.nil_append,
    ComputeKernel.inputs, ComputeKernel.outputs,
    ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType,
    Bool.false_eq_true, ↓reduceIte]
  all_goals rfl

/-- All storage annotations are erased on the real specification surface only. -/
theorem forward_log_projection (pred : Storage) (target : MathStorage) (output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat) :
    (forwardLog pred target output v Y T O sy st so N B eps reduction).eraseDType.toAlgKernel =
      forwardMath v Y T O sy st so N B eps Bool.true reduction := by
  simp only [ComputeKernel.eraseDType, forward_log_projects,
    ComputeKernel.toAlgKernel_fromKernelBody]
  cases pred <;> cases target <;> cases output <;> cases v
  all_goals simp only [forwardLog, forwardLogOriginal, commuteOutput, forwardMath, rowPointer, forwardPrefix, FullAlgorithm.forward, FullAlgorithm.forwardSurface, ComputeKernel.eraseDType, ComputeKernel.fromKernelBody, forwardLog_fp32_fp32, forwardLog_fp32_fp64, forwardLog_fp64_fp32, forwardLog_fp64_fp64, forwardLog_fp16_fp32, forwardLog_fp16_fp64, forwardLog_bf16_fp32, forwardLog_bf16_fp64, forwardLog_f8e4_fp32, forwardLog_f8e4_fp64, forwardLog_f8e5_fp32, forwardLog_f8e5_fp64,
    List.take, List.drop, List.cons_append, List.nil_append,
    ComputeKernel.inputs, ComputeKernel.outputs,
    ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType,
    Bool.false_eq_true, ↓reduceIte]
  all_goals rfl

/-- All storage annotations are erased on the real specification surface only. -/
theorem backward_default_projection (target output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) :
    (backwardDefault target output v T O st so N B).eraseDType.toAlgKernel =
      backwardMath v T O st so N B Bool.false := by
  simp only [ComputeKernel.eraseDType, backward_default_projects,
    ComputeKernel.toAlgKernel_fromKernelBody]
  cases target <;> cases output <;> cases v
  all_goals simp only [backwardDefault, backwardDefaultOriginal, commuteOutput, backwardMath, rowPointer, backwardPrefix, FullAlgorithm.backward, FullAlgorithm.backwardSurface,
    List.take, List.drop, List.cons_append, List.nil_append,
    ComputeKernel.inputs, ComputeKernel.outputs,
    ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType,
    Bool.false_eq_true, ↓reduceIte]
  all_goals rfl

/-- All storage annotations are erased on the real specification surface only. -/
theorem backward_log_projection (target : MathStorage) (output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) :
    (backwardLog target output v T O st so N B).eraseDType.toAlgKernel =
      backwardMath v T O st so N B Bool.true := by
  simp only [ComputeKernel.eraseDType, backward_log_projects,
    ComputeKernel.toAlgKernel_fromKernelBody]
  cases target <;> cases output <;> cases v
  all_goals simp only [backwardLog, backwardLogOriginal, commuteOutput, backwardMath, rowPointer, backwardPrefix, FullAlgorithm.backward, FullAlgorithm.backwardSurface,
    List.take, List.drop, List.cons_append, List.nil_append,
    ComputeKernel.inputs, ComputeKernel.outputs,
    ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType,
    Bool.false_eq_true, ↓reduceIte]
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KldivOps.Source
