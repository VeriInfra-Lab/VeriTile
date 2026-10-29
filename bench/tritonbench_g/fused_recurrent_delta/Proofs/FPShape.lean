import bench.tritonbench_g.fused_recurrent_delta.Proofs.FPStages
import bench.tritonbench_g.fused_recurrent_hgrn.Proofs.FPShape
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentDelta.FPStages
open VeriTile.Triton FP.Structural
open FusedRecurrentHgrn.FPStages (erase_real_cast)
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

theorem output_stage_shape (HNext q o : RegionName) (t s_qk_h s_vo_h B H K V BK BV : Nat) (scale : ℝ) :
    (output HNext q o t s_qk_h s_vo_h B H K V BK BV scale).eraseDType.surfaceBody =
      (fused_recurrent_delta_output_step_slice HNext q o t s_qk_h s_vo_h B H K V BK BV scale).eraseDType.surfaceBody := by
  simp [output, fused_recurrent_delta_output_step_slice, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    ComputeKernel.fromKernelBody, ComputeKernel.surfaceBody, ComputeDType.eraseDType,
    Except.bind, bind, pure, erase_real_cast]
  repeat' constructor
  all_goals conv_lhs => rw [Op.eraseDType.eq_def]
  all_goals simp [Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, VeriTile.Triton.eraseDType]

theorem vnew_stage_shape (HPrev k v : RegionName) (t s_qk_h s_vo_h K V BK BV : Nat) :
    (vnew HPrev k v t s_qk_h s_vo_h K V BK BV).eraseDType.surfaceBody =
      (fused_recurrent_delta_vnew_step_slice HPrev k v t s_qk_h s_vo_h K V BK BV).eraseDType.surfaceBody := by
  simp [vnew, fused_recurrent_delta_vnew_step_slice, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    ComputeKernel.fromKernelBody, ComputeKernel.surfaceBody, ComputeDType.eraseDType,
    Except.bind, bind, pure, erase_real_cast]
  repeat' constructor
  all_goals conv_lhs => rw [Op.eraseDType.eq_def]
  all_goals simp [Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, VeriTile.Triton.eraseDType]

#axiomsClean output_stage_shape
#axiomsClean vnew_stage_shape
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentDelta.FPStages
