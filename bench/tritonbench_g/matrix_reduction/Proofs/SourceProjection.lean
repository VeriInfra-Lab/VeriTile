import bench.tritonbench_g.matrix_reduction.Kernels
import bench.tritonbench_g.matrix_reduction.Proofs.LegacyBody

/-! Full algorithm equality to the frozen independent mathematical companion.
This is only the Real projection; FP proofs retain typed loads and casts. -/
namespace VeriTile.Bench.TritonBenchG.MatrixReduction
open VeriTile.Triton
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

theorem mathematical_supported (X Y : RegionName) (sm sn sy M N : Nat) :
    (load_reduce_kernel X Y sm sn sy M N).toAlgorithm? =
      .ok (load_reduce_kernel X Y sm sn sy M N).toAlgKernel := by
  have h : (TypedSource.source32 X Y sm sn sy M N).toAlgorithm? =
      .ok (TypedSource.source32 X Y sm sn sy M N).toAlgKernel := by rfl
  simp only [load_reduce_kernel,ComputeKernel.eraseDType,h,
    ComputeKernel.toAlgorithm?_fromKernelBody,ComputeKernel.toAlgKernel_fromKernelBody]

theorem mathematical_projection (X Y : RegionName) (sm sn sy M N : Nat) :
    (load_reduce_kernel X Y sm sn sy M N).toAlgKernel =
      (LegacyBody.load_reduce_kernel X Y sm sn sy M N).toAlgKernel := by
  simp [load_reduce_kernel,TypedSource.source32,LegacyBody.load_reduce_kernel,
    ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,Except.bind,bind,Kernel.eraseDType,Stmt.eraseDTypeList,
    Stmt.eraseDType,Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,
    ComputeKernel.fromKernelBody,ComputeDType.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType]

end VeriTile.Bench.TritonBenchG.MatrixReduction
