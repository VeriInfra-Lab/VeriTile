import bench.repairs.tritonbench_g.softmax_flaggems.Proofs.ZeroPadding
import bench.tritonbench_g.softmax_flaggems.Proofs.HostGridHost
import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardInnerFormula
import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardNonInnerFormula

namespace VeriTile.Bench.Repairs.SoftmaxFlaggems.SourceBridge
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

macro "zero_padding_source" : tactic => `(tactic|
  repeat' first
  | exact ZeroPadding.refl _
  | apply ZeroPadding.cons
  | apply ZeroPadding.branch
  | apply ZeroPadding.dynamic_range
  | apply ZeroPadding.static_range
  | apply ZeroPadding.assign
  | exact ZeroPadding.load_ptr _ _ _ _ _ ‹ZeroPadding.Zero _›
  | exact ZeroPadding.load_region _ _ _ _ _ _ ‹ZeroPadding.Zero _›
  | intro s hs
  | solve | simp [Stmt.FlattenOk, StmtList.FlattenOk, Op.FlattenOk.eq_def,
      MemAccess.FlattenOkC, MaskOpt.FlattenOkC, MemAccess.eraseDType, MaskOpt.eraseDType,
      FloatDType.toTileDType]
  | rfl)

/-- Every changed load, in both source branches and both source loops, is an
explicit zero load. This is a whole-program execution equality on the old
zero-sampler interpretation, not a numerical result premise. -/
theorem inner_body (P G Y : RegionName) (M N C B : Nat) (one : Bool) :
    ZeroPadding.Equivalent
      (FullSource.softmax_backward_kernel_inner .fp32 P G Y M N C B one).eraseDType.toAlgKernel.body
      (Source.softmax_backward_kernel_inner_fp32 P G Y M N C B one).eraseDType.toAlgKernel.body := by
  simp only [FullSource.softmax_backward_kernel_inner, Source.softmax_backward_kernel_inner_fp32,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind,
    MemAccess.eraseDType, MaskOpt.eraseDType, FloatDType.toTileDType]
  zero_padding_source

theorem noninner_body (P G Y : RegionName) (M N K B C : Nat) (one : Bool) :
    ZeroPadding.Equivalent
      (FullSource.softmax_backward_kernel_non_inner .fp32 P G Y M N K B C one).eraseDType.toAlgKernel.body
      (Source.softmax_backward_kernel_non_inner_fp32 P G Y M N K B C one).eraseDType.toAlgKernel.body := by
  simp only [FullSource.softmax_backward_kernel_non_inner, Source.softmax_backward_kernel_non_inner_fp32,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind,
    MemAccess.eraseDType, MaskOpt.eraseDType, FloatDType.toTileDType]
  zero_padding_source

#axiomsClean inner_body
#axiomsClean noninner_body
end VeriTile.Bench.Repairs.SoftmaxFlaggems.SourceBridge
