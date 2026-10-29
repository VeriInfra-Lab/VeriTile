import bench.tritonbench_g.logsumexp_fwd.Proofs.FPStages
import bench.tritonbench_g.fused_recurrent_hgrn.Proofs.FPShape
namespace VeriTile.Bench.TritonBenchG.LogsumexpFwd.FPStages
open VeriTile.Triton FP.Structural
open FusedRecurrentHgrn.FPStages (erase_real_cast)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 15000
set_option linter.unusedSimpArgs false

theorem row_block_shape (x z : RegionName) (D B : Nat) (HAS_SCALE : Bool) (scale : ℝ) :
    (rowBlock x z D B HAS_SCALE scale).eraseDType.surfaceBody = (logsumexp_fwd_kernel x z D B HAS_SCALE scale).eraseDType.surfaceBody := by
  simp [rowBlock, logsumexp_fwd_kernel, ComputeKernel.eraseDType,
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
#axiomsClean row_block_shape
end
end VeriTile.Bench.TritonBenchG.LogsumexpFwd.FPStages
