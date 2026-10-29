import bench.tritonbench_g.fused_recurrent_delta.Proofs.FPBackwardKernels
import bench.tritonbench_g.fused_recurrent_hgrn.Proofs.FPShape
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentDelta.FPStages
open VeriTile.Triton FP.Structural
open FusedRecurrentHgrn.FPStages (erase_real_cast)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 15000
set_option linter.unusedSimpArgs false

theorem dkHeadwise_stage_shape (DHPrev q do_ v beta dk : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) :
    (dkHeadwise DHPrev q do_ v beta dk t s_qk_h s_vo_h B H T K V BK BV scale).eraseDType.surfaceBody =
      (fused_recurrent_delta_bwd_dk_step_slice_headwise DHPrev q do_ v beta dk t s_qk_h s_vo_h B H T K V BK BV scale).eraseDType.surfaceBody := by
  simp [dkHeadwise, fused_recurrent_delta_bwd_dk_step_slice_headwise, ComputeKernel.eraseDType,
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

#axiomsClean dkHeadwise_stage_shape

theorem dkScalar_stage_shape (DHPrev q do_ v beta dk : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) :
    (dkScalar DHPrev q do_ v beta dk t s_qk_h s_vo_h B H T K V BK BV scale).eraseDType.surfaceBody =
      (fused_recurrent_delta_bwd_dk_step_slice_scalarbeta DHPrev q do_ v beta dk t s_qk_h s_vo_h B H T K V BK BV scale).eraseDType.surfaceBody := by
  simp [dkScalar, fused_recurrent_delta_bwd_dk_step_slice_scalarbeta, ComputeKernel.eraseDType,
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

#axiomsClean dkScalar_stage_shape

theorem dvHeadwise_stage_shape (DHPrev q do_ k beta dv : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) :
    (dvHeadwise DHPrev q do_ k beta dv t s_qk_h s_vo_h B H T K V BK BV scale).eraseDType.surfaceBody =
      (fused_recurrent_delta_bwd_dv_step_slice_headwise DHPrev q do_ k beta dv t s_qk_h s_vo_h B H T K V BK BV scale).eraseDType.surfaceBody := by
  simp [dvHeadwise, fused_recurrent_delta_bwd_dv_step_slice_headwise, ComputeKernel.eraseDType,
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

#axiomsClean dvHeadwise_stage_shape

theorem dvScalar_stage_shape (DHPrev q do_ k beta dv : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) :
    (dvScalar DHPrev q do_ k beta dv t s_qk_h s_vo_h B H T K V BK BV scale).eraseDType.surfaceBody =
      (fused_recurrent_delta_bwd_dv_step_slice_scalarbeta DHPrev q do_ k beta dv t s_qk_h s_vo_h B H T K V BK BV scale).eraseDType.surfaceBody := by
  simp [dvScalar, fused_recurrent_delta_bwd_dv_step_slice_scalarbeta, ComputeKernel.eraseDType,
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

#axiomsClean dvScalar_stage_shape

theorem dbetaHeadwise_stage_shape (DHPrev q do_ k v dbeta : RegionName)
    (t s_qk_h s_vo_h NK B H T K V BK BV : Nat) (scale : ℝ) :
    (dbetaHeadwise DHPrev q do_ k v dbeta t s_qk_h s_vo_h NK B H T K V BK BV scale).eraseDType.surfaceBody =
      (fused_recurrent_delta_bwd_dbeta_step_slice_headwise DHPrev q do_ k v dbeta t s_qk_h s_vo_h NK B H T K V BK BV scale).eraseDType.surfaceBody := by
  simp [dbetaHeadwise, fused_recurrent_delta_bwd_dbeta_step_slice_headwise, ComputeKernel.eraseDType,
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

#axiomsClean dbetaHeadwise_stage_shape

theorem dbetaScalar_stage_shape (DHPrev q do_ k v dbeta : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) :
    (dbetaScalar DHPrev q do_ k v dbeta t s_qk_h s_vo_h B H T K V BK BV scale).eraseDType.surfaceBody =
      (fused_recurrent_delta_bwd_dbeta_step_slice_scalarbeta DHPrev q do_ k v dbeta t s_qk_h s_vo_h B H T K V BK BV scale).eraseDType.surfaceBody := by
  simp [dbetaScalar, fused_recurrent_delta_bwd_dbeta_step_slice_scalarbeta, ComputeKernel.eraseDType,
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

#axiomsClean dbetaScalar_stage_shape

theorem dqHeadwise_stage_shape (HPrev k v beta do_ dq : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) :
    (dqHeadwise HPrev k v beta do_ dq t s_qk_h s_vo_h B H T K V BK BV scale).eraseDType.surfaceBody =
      (fused_recurrent_delta_bwd_dq_step_slice_headwise HPrev k v beta do_ dq t s_qk_h s_vo_h B H T K V BK BV scale).eraseDType.surfaceBody := by
  simp [dqHeadwise, fused_recurrent_delta_bwd_dq_step_slice_headwise, ComputeKernel.eraseDType,
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

#axiomsClean dqHeadwise_stage_shape

theorem dqScalar_stage_shape (HPrev k v beta do_ dq : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) :
    (dqScalar HPrev k v beta do_ dq t s_qk_h s_vo_h B H T K V BK BV scale).eraseDType.surfaceBody =
      (fused_recurrent_delta_bwd_dq_step_slice_scalarbeta HPrev k v beta do_ dq t s_qk_h s_vo_h B H T K V BK BV scale).eraseDType.surfaceBody := by
  simp [dqScalar, fused_recurrent_delta_bwd_dq_step_slice_scalarbeta, ComputeKernel.eraseDType,
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

#axiomsClean dqScalar_stage_shape

theorem dkCorrection_stage_shape (HPrev dv dk : RegionName) (t s_qk_h s_vo_h B H K V BK BV : Nat) :
    (dkCorrection HPrev dv dk t s_qk_h s_vo_h B H K V BK BV).eraseDType.surfaceBody =
      (fused_recurrent_delta_bwd_dk_correction_step_slice HPrev dv dk t s_qk_h s_vo_h B H K V BK BV).eraseDType.surfaceBody := by
  simp [dkCorrection, fused_recurrent_delta_bwd_dk_correction_step_slice, ComputeKernel.eraseDType,
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

#axiomsClean dkCorrection_stage_shape
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentDelta.FPStages
