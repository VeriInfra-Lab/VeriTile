import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardSource
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardRun
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardTraceControl

namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward
open VeriTile.Triton
set_option maxHeartbeats 800000
set_option maxRecDepth 16000

/-- A single complete causal log profile tests the trace-safety bridge without
evaluating arithmetic or precomputing an output state. -/
theorem fp16_log_causal_trace_probe (GradIn GradOut Out : RegionName)
    (bm bn gm gn om on K D : Nat) (bounds : RegionBounds) (s : BlockState) :
    Kernel.TraceSafe bounds (Typed.homogeneousKernel .fp16 Bool.false GradIn GradOut Out
      bm bn gm gn om on K D Bool.true Bool.true Bool.true).eraseDType.toAlgKernel s ↔
    Kernel.TraceSafe bounds (source GradIn GradOut Out bm bn gm gn om on K D
      Bool.true Bool.true Bool.true).toAlgKernel s := by
  have hp : (Typed.homogeneousKernel .fp16 Bool.false GradIn GradOut Out
      bm bn gm gn om on K D Bool.true Bool.true Bool.true).toAlgorithm? =
      .ok (Typed.homogeneousKernel .fp16 Bool.false GradIn GradOut Out
        bm bn gm gn om on K D Bool.true Bool.true Bool.true).toAlgKernel := by rfl
  simp only [ComputeKernel.eraseDType, hp, ComputeKernel.toAlgKernel_fromKernelBody]
  dsimp only [Typed.homogeneousKernel, source, ksoftmax_backward_surface,
    ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Except.bind, Except.pure, Bind.bind, Pure.pure, Kernel.eraseDType,
    ComputeDType.eraseDType]
  simp only [Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType,
    MemAccess.eraseDType, MaskOpt.eraseDType, VeriTile.Triton.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    FloatDType.toTileDType, Region.cast]
  simp only [Op.eraseDType_castFloat, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, FloatDType.toTileDType, Region.cast]
  simp only [Kernel.TraceSafe, Stmt.TraceSafeList,
    StaticControl.trace_if_true, StaticControl.trace_if_false,
    StaticControl.trace_ifElse_true, StaticControl.trace_ifElse_false,
    StaticControl.trace_assign_real_cast, List.nil_append, List.cons_append,
    StaticControl.cons_bind, stepStmts.nil,
    StaticControl.if_true, StaticControl.if_false, StaticControl.assign_real_cast,
    StaticControl.ifElse_true, StaticControl.ifElse_false,
    Option.bind_assoc, Option.bind_some, Option.bind_none, Option.bind_fun_some]
  rfl

#axiomsClean fp16_log_causal_trace_probe
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward
