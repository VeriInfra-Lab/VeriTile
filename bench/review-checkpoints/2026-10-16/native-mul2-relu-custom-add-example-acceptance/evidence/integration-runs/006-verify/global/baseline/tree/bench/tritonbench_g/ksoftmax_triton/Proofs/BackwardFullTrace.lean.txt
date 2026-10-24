import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardFullErasure
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardTraceControl

namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward
open VeriTile.Triton
set_option maxHeartbeats 16000000
set_option maxRecDepth 16000

/-- Trace safety of the complete independently typed source after explicit
dtype erasure. This separately proves safety preservation; equal final states
alone would not justify transport of bounds obligations. -/
theorem full_typed_erased_trace_equiv (gradient saved destination : Typed.Storage)
    (GradIn GradOut Out : RegionName) (bm bn gm gn om on K D : Nat)
    (log causal fp16 : Bool) (bounds : RegionBounds) (s : BlockState) :
    Kernel.TraceSafe bounds (Typed.kernel gradient saved destination Bool.false GradIn GradOut Out
      bm bn gm gn om on K D log causal fp16).eraseDType.toAlgKernel s ↔
    Kernel.TraceSafe bounds (source GradIn GradOut Out bm bn gm gn om on K D log causal fp16).toAlgKernel s := by
  simp only [ComputeKernel.eraseDType, full_typed_projects,
    ComputeKernel.toAlgKernel_fromKernelBody]
  cases gradient <;> cases saved <;> cases destination <;>
    cases log <;> cases causal <;> cases fp16
  all_goals
    dsimp only [Typed.kernel,
      Typed.kernel_fp32_fp32, Typed.kernel_fp32_fp64, Typed.kernel_fp32_fp16,
      Typed.kernel_fp32_bf16, Typed.kernel_fp32_f8e4, Typed.kernel_fp32_f8e5,
      Typed.kernel_fp64_fp32, Typed.kernel_fp64_fp64, Typed.kernel_fp64_fp16,
      Typed.kernel_fp64_bf16, Typed.kernel_fp64_f8e4, Typed.kernel_fp64_f8e5,
      Typed.kernel_fp16_fp32, Typed.kernel_fp16_fp64, Typed.kernel_fp16_fp16,
      Typed.kernel_fp16_bf16, Typed.kernel_fp16_f8e4, Typed.kernel_fp16_f8e5,
      Typed.kernel_bf16_fp32, Typed.kernel_bf16_fp64, Typed.kernel_bf16_fp16,
      Typed.kernel_bf16_bf16, Typed.kernel_bf16_f8e4, Typed.kernel_bf16_f8e5,
      Typed.kernel_f8e4_fp32, Typed.kernel_f8e4_fp64, Typed.kernel_f8e4_fp16,
      Typed.kernel_f8e4_bf16, Typed.kernel_f8e4_f8e4, Typed.kernel_f8e4_f8e5,
      Typed.kernel_f8e5_fp32, Typed.kernel_f8e5_fp64, Typed.kernel_f8e5_fp16,
      Typed.kernel_f8e5_bf16, Typed.kernel_f8e5_f8e4, Typed.kernel_f8e5_f8e5,
      source, ksoftmax_backward_surface,
      ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, Except.bind, Except.pure, Bind.bind, Pure.pure,
      Kernel.eraseDType, ComputeDType.eraseDType]
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

#axiomsClean full_typed_erased_trace_equiv
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward
