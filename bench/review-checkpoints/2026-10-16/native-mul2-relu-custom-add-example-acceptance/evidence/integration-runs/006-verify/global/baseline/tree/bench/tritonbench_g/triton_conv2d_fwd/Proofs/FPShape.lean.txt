import bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPTypedSource
import bench.tritonbench_g.triton_conv2d_fwd.TritonConv2dFwd
namespace VeriTile.Bench.TritonBenchG.TritonConv2dFwd.FPValue
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false

def legacySource (p : Parameters) : ComputeKernel :=
  conv2d_forward_surface p.Input p.Weight p.Output
    p.batch_dim p.in_feat_dim p.in_height p.in_width p.out_feat_dim p.out_height p.out_width p.input_batch_stride p.input_in_feat_stride p.input_height_stride p.input_width_stride p.weight_out_feat_stride p.weight_in_feat_stride p.weight_height_stride p.weight_width_stride p.output_batch_stride p.output_out_feat_stride p.output_height_stride p.output_width_stride p.kernel_height p.kernel_width p.stride_height p.stride_width p.padding_height p.padding_width p.groups Bool.true p.tf32
    p.BLOCK_SIZE_BATCH_HEIGHT_WIDTH p.BLOCK_SIZE_IN_FEAT p.BLOCK_SIZE_OUT_FEAT

/-- Exact original control, address, mask, dot options and full triple-loop
projection. Dtype erasure here is syntactic; no floating value erasure is used. -/
theorem source_projection (p : Parameters) :
    (source p).eraseDType.surfaceBody = (legacySource p).eraseDType.surfaceBody := by
  simp [source, legacySource, typed, conv2d_forward_surface, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    ComputeKernel.fromKernelBody, ComputeKernel.surfaceBody, ComputeDType.eraseDType,
    Except.bind, bind, pure, FloatDType.toTileDType, FloatDType.numericDType]
  all_goals repeat' constructor
  all_goals try conv_lhs => rw [Op.eraseDType.eq_def]
  all_goals try simp [Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, VeriTile.Triton.eraseDType]
  all_goals simp only [FloatDType.toTileDType, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, Op.eraseDType.eq_def, NumericDType.eraseDType,
    ComparableDType.eraseDType, VeriTile.Triton.eraseDType]
  all_goals try simp [eq_mp_eq_cast, eq_mpr_eq_cast, cast_eq, id_eq]
  all_goals rfl
#axiomsClean source_projection
end
end VeriTile.Bench.TritonBenchG.TritonConv2dFwd.FPValue
