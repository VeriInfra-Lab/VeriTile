import bench.tritonbench_g.fused_recurrent_hgrn.Proofs.FPStages
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000
@[simp] theorem erase_real_cast {shape : TileShape} (e : Op .real shape) :
    (Op.castFloat .real .real e).eraseDType = e.eraseDType := by
  rw [Op.eraseDType_castFloat]
  rfl

theorem forward_stage_shape (BHPrev X G O : RegionName) (t T D BD : Nat) :
    (forward BHPrev X G O t T D BD).eraseDType.surfaceBody =
      (fused_recurrent_hgrn_forward_step_store_slice BHPrev X G O t T D BD).eraseDType.surfaceBody := by
  simp [forward, fused_recurrent_hgrn_forward_step_store_slice, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    ComputeKernel.fromKernelBody, ComputeKernel.surfaceBody, ComputeDType.eraseDType,
    Except.bind, bind, pure, erase_real_cast]
  repeat' constructor
  all_goals conv_lhs => rw [Op.eraseDType.eq_def]
  all_goals simp [Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, VeriTile.Triton.eraseDType]

theorem dx_stage_shape (DHPrev DO DX : RegionName) (t T D BD : Nat) :
    (dx DHPrev DO DX t T D BD).eraseDType.surfaceBody =
      (fused_recurrent_hgrn_bwd_dx_step_store_slice DHPrev DO DX t T D BD).eraseDType.surfaceBody := by
  simp [dx, fused_recurrent_hgrn_bwd_dx_step_store_slice, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    ComputeKernel.fromKernelBody, ComputeKernel.surfaceBody, ComputeDType.eraseDType,
    Except.bind, bind, pure, erase_real_cast]
  repeat' constructor
  all_goals conv_lhs => rw [Op.eraseDType.eq_def]
  all_goals simp [Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, VeriTile.Triton.eraseDType]

theorem dxTwo_stage_shape (DHPrev DO G DX : RegionName) (t T D BD : Nat) :
    (dxTwo DHPrev DO G DX t T D BD).eraseDType.surfaceBody =
      (fused_recurrent_hgrn_bwd_dx_two_step_store_slice DHPrev DO G DX t T D BD).eraseDType.surfaceBody := by
  simp [dxTwo, fused_recurrent_hgrn_bwd_dx_two_step_store_slice, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    ComputeKernel.fromKernelBody, ComputeKernel.surfaceBody, ComputeDType.eraseDType,
    Except.bind, bind, pure, erase_real_cast]
  repeat' constructor
  all_goals conv_lhs => rw [Op.eraseDType.eq_def]
  all_goals simp [Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, VeriTile.Triton.eraseDType]

theorem dg_stage_shape (DHPrev DO O H0 DG : RegionName) (U : Bool) (t T D BD : Nat) :
    (dg DHPrev DO O H0 DG U t T D BD).eraseDType.surfaceBody =
      (fused_recurrent_hgrn_bwd_dg_step_store_slice DHPrev DO O H0 DG U t T D BD).eraseDType.surfaceBody := by
  simp [dg, fused_recurrent_hgrn_bwd_dg_step_store_slice, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    ComputeKernel.fromKernelBody, ComputeKernel.surfaceBody, ComputeDType.eraseDType,
    Except.bind, bind, pure, erase_real_cast]
  repeat' constructor
  all_goals conv_lhs => rw [Op.eraseDType.eq_def]
  all_goals simp [Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, VeriTile.Triton.eraseDType]
#axiomsClean forward_stage_shape
#axiomsClean dx_stage_shape
#axiomsClean dxTwo_stage_shape
#axiomsClean dg_stage_shape
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages
