import bench.tritonbench_g.parallel_attention.Proofs.FPBackwardWhole
import bench.tritonbench_g.parallel_attention.ParallelAttention
import bench.tritonbench_g.fused_recurrent_hgrn.Proofs.FPShape
namespace VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
open VeriTile.Triton FP.Structural
open FusedRecurrentHgrn.FPStages (erase_real_cast)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem forward_shape (c : ForwardConfig) :
    (forwardKernel c).eraseDType.surfaceBody =
      (pa_fwd_surface c.q c.k c.v c.o c.z c.s_qk_h c.s_qk_t c.s_qk_d
        c.s_vo_h c.s_vo_t c.s_vo_d c.scale c.B c.H c.T c.K c.V c.BTL c.BTS c.BK c.BV).eraseDType.surfaceBody := by

  simp [forwardKernel, forwardSource, pa_fwd_surface, ComputeKernel.eraseDType,
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

theorem backward_shape (c : BackwardConfig) :
    (backwardKernel c).eraseDType.surfaceBody =
      (pa_bwd_dkv_surface c.q c.k c.v c.do_ c.dz c.dk c.dv c.i_bh c.i_c c.i_k c.i_v c.i_h
        c.s_qk_h c.s_qk_t c.s_qk_d c.s_vo_h c.s_vo_t c.s_vo_d c.scale
        c.B c.H c.T c.K c.V c.BTL c.BTS c.BK c.BV).eraseDType.surfaceBody := by

  simp [backwardKernel, backwardSource, pa_bwd_dkv_surface, ComputeKernel.eraseDType,
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

#axiomsClean forward_shape
#axiomsClean backward_shape
end
end VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
