import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.layer_norm_liger.Proofs.FPSupplied
import bench.tritonbench_g.layer_norm_liger.Proofs.FPTypedSource
import bench.tritonbench_g.layer_norm_liger.LayerNormLiger
namespace VeriTile.Bench.TritonBenchG.LayerNormLiger.FPStages
open VeriTile.Triton
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
@[simp] theorem erase_real_cast (e : Op .real shape) :
    (Op.castFloat .real .real e).eraseDType = e.eraseDType :=
  Op.eraseDType_castFloat .real .real e
/-- Source structure correspondence to the complete original accepted kernel.
This is a type-erasure bridge, not an assertion that erasing rounding preserves
floating values. The execution/value theorem uses the typed kernel itself. -/
theorem source_shape (p : Parameters) :
    (kernel p).eraseDType.surfaceBody =
      (layer_norm_liger_forward_surface p.y p.yStride p.x p.xStride
        p.weight p.weightStride p.bias p.biasStride p.mean p.meanStride
        p.rstd p.rstdStride p.columns p.epsilon p.block).eraseDType.surfaceBody := by
  simp [kernel, source, layer_norm_liger_forward_surface, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    ComputeKernel.fromKernelBody, ComputeKernel.surfaceBody, ComputeDType.eraseDType,
    Except.bind, bind, pure, FloatDType.toTileDType, FloatDType.numericDType]
  all_goals repeat' constructor
  all_goals conv_lhs => rw [Op.eraseDType.eq_def]
  all_goals simp [Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, VeriTile.Triton.eraseDType]
  all_goals simp only [FloatDType.toTileDType, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, Op.eraseDType.eq_def, NumericDType.eraseDType,
    ComparableDType.eraseDType, VeriTile.Triton.eraseDType]
  all_goals try simp [eq_mp_eq_cast, eq_mpr_eq_cast, cast_eq, id_eq]

theorem mean_store_shape (pre output : RegionName) (stride : Nat) :
    (meanStore .fp32 pre output stride).eraseDType.surfaceBody =
      (layer_norm_liger_forward_mean_store_slice pre output stride).eraseDType.surfaceBody := by
  simp [meanStore, layer_norm_liger_forward_mean_store_slice, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    ComputeKernel.fromKernelBody, ComputeKernel.surfaceBody, ComputeDType.eraseDType,
    Except.bind, bind, pure, FloatDType.toTileDType, FloatDType.numericDType]
  all_goals repeat' constructor
  all_goals conv_lhs => rw [Op.eraseDType.eq_def]
  all_goals simp [Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, VeriTile.Triton.eraseDType]
  all_goals simp only [FloatDType.toTileDType, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, Op.eraseDType.eq_def, NumericDType.eraseDType,
    ComparableDType.eraseDType, VeriTile.Triton.eraseDType]
  all_goals try simp [eq_mp_eq_cast, eq_mpr_eq_cast, cast_eq, id_eq]

#axiomsClean mean_store_shape

theorem rstd_store_shape (pre output : RegionName) (stride : Nat) :
    (rstdStore .fp32 pre output stride).eraseDType.surfaceBody =
      (layer_norm_liger_forward_rstd_store_slice pre output stride).eraseDType.surfaceBody := by
  simp [rstdStore, layer_norm_liger_forward_rstd_store_slice, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    ComputeKernel.fromKernelBody, ComputeKernel.surfaceBody, ComputeDType.eraseDType,
    Except.bind, bind, pure, FloatDType.toTileDType, FloatDType.numericDType]
  all_goals repeat' constructor
  all_goals conv_lhs => rw [Op.eraseDType.eq_def]
  all_goals simp [Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, VeriTile.Triton.eraseDType]
  all_goals simp only [FloatDType.toTileDType, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, Op.eraseDType.eq_def, NumericDType.eraseDType,
    ComparableDType.eraseDType, VeriTile.Triton.eraseDType]
  all_goals try simp [eq_mp_eq_cast, eq_mpr_eq_cast, cast_eq, id_eq]

#axiomsClean rstd_store_shape

#axiomsClean source_shape
end VeriTile.Bench.TritonBenchG.LayerNormLiger.FPStages
