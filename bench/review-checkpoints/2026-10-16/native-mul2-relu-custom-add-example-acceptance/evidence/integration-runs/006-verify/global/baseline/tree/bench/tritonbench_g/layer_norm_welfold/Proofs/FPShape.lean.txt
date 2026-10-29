import bench.tritonbench_g.layer_norm_welfold.Proofs.FPSourceBlock
import bench.tritonbench_g.layer_norm_welfold.Proofs.FPStages
import bench.tritonbench_g.fused_recurrent_hgrn.Proofs.FPShape
namespace VeriTile.Bench.TritonBenchG.LayerNormWelfold.FPStages
open VeriTile.Triton FP.Structural
open FusedRecurrentHgrn.FPStages (erase_real_cast)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 15000
set_option linter.unusedSimpArgs false

theorem normalize_chunk_shape (mean rstd X W Bias Y : RegionName) (rnumel RBLOCK : Nat) :
    (normalizeChunk mean rstd X W Bias Y rnumel RBLOCK).eraseDType.surfaceBody = (layer_norm_welfold_normalize_slice mean rstd X W Bias Y rnumel RBLOCK).eraseDType.surfaceBody := by
  simp [normalizeChunk, layer_norm_welfold_normalize_slice, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    ComputeKernel.fromKernelBody, ComputeKernel.surfaceBody, ComputeDType.eraseDType,
    Except.bind, bind, pure, erase_real_cast]
  repeat' constructor
  all_goals conv_lhs => rw [Op.eraseDType.eq_def]
  all_goals simp [Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, VeriTile.Triton.eraseDType]
  all_goals repeat' constructor
  all_goals rfl
#axiomsClean normalize_chunk_shape

theorem source_one_block_shape (mean rstd X W Bias Y : RegionName) (xnumel rnumel RBLOCK : Nat) :
    (sourceOneBlock mean rstd X W Bias Y xnumel rnumel RBLOCK).eraseDType.surfaceBody =
      (layer_norm_welfold_surface mean rstd X W Bias Y xnumel rnumel 1 RBLOCK).eraseDType.surfaceBody := by
  simp [sourceOneBlock, sourceTyped, layer_norm_welfold_surface, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    ComputeKernel.fromKernelBody, ComputeKernel.surfaceBody, ComputeDType.eraseDType,
    Except.bind, bind, pure, erase_real_cast]
  repeat' constructor
  all_goals conv_lhs => rw [Op.eraseDType.eq_def]
  all_goals simp [Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, VeriTile.Triton.eraseDType]
  all_goals repeat' constructor
  all_goals rfl
#axiomsClean source_one_block_shape

end
end VeriTile.Bench.TritonBenchG.LayerNormWelfold.FPStages
