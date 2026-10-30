import bench.tritonbench_g.fused_rwkv6_kernel.Proofs.FPStages
import bench.tritonbench_g.fused_recurrent_delta.Proofs.FPStateShape
namespace VeriTile.Bench.TritonBenchG.FusedRwkv6Kernel.FPStages
open VeriTile.Triton FP.Structural
open FusedRecurrentHgrn.FPStages (erase_real_cast)
open FusedRecurrentDelta.FPStages (selectBranches selectBranch)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 15000
set_option linter.unusedSimpArgs false

theorem state_stage_shape (BHPrev k v w BHOut : RegionName) (t s_k_h s_v_h K V BK BV : Nat) :
    (stateStep BHPrev k v w BHOut t s_k_h s_v_h K V BK BV).eraseDType.surfaceBody = (fused_recurrent_rwkv6_state_step_slice BHPrev k v w BHOut t s_k_h s_v_h K V BK BV).eraseDType.surfaceBody := by
  simp [stateStep, fused_recurrent_rwkv6_state_step_slice, ComputeKernel.eraseDType,
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

#axiomsClean state_stage_shape

theorem output_stage_shape (BHPrev q k v u o : RegionName) (t s_k_h s_v_h B H T K V BK BV : Nat) (scale : ℝ) :
    (outputStep BHPrev q k v u o t s_k_h s_v_h B H T K V BK BV scale).eraseDType.surfaceBody = (fused_recurrent_rwkv6_output_step_slice BHPrev q k v u o t s_k_h s_v_h B H T K V BK BV scale).eraseDType.surfaceBody := by
  simp [outputStep, fused_recurrent_rwkv6_output_step_slice, ComputeKernel.eraseDType,
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

#axiomsClean output_stage_shape

def selectFinalBranch (st : ComputeStmt) : List ComputeStmt :=
  match st with
  | .ifThen (.constBool b) yes => if b then yes else []
  | _ => [st]

def selectFinalBranches (body : List ComputeStmt) : List ComputeStmt := body.flatMap selectFinalBranch

theorem selectFinalBranch_exec {α : Type} [Inhabited α] (M : Algebra α) (st : ComputeStmt) (s : State α) :
    run M (selectFinalBranch st) s = step M st s := by
  unfold selectFinalBranch
  split
  · rename_i b yes
    cases b <;> simp [step, run, evalOp_unfold, Option.bind]
  · simp [run]

theorem selectFinalBranches_exec {α : Type} [Inhabited α] (M : Algebra α) (body : List ComputeStmt) (s : State α) :
    run M (selectFinalBranches body) s = run M body s := by
  induction body generalizing s with
  | nil => rfl
  | cons st rest ih =>
    simp only [selectFinalBranches, List.flatMap_cons, run_append, selectFinalBranch_exec, run]
    simp only [selectFinalBranches] at ih
    cases h : step M st s <;> simp [h, ih]

#axiomsClean selectFinalBranch_exec
#axiomsClean selectFinalBranches_exec

theorem final_enabled_stage_shape (BHPrev k v w ht : RegionName) (t s_k_h s_v_h K V BK BV : Nat) :
    (ComputeKernel.mk [BHPrev,k,v,w] [ht] (selectFinalBranches (finalStep BHPrev k v w ht Bool.true t s_k_h s_v_h K V BK BV).surfaceBody)).eraseDType.surfaceBody = (fused_recurrent_rwkv6_state_step_slice BHPrev k v w ht t s_k_h s_v_h K V BK BV).eraseDType.surfaceBody := by
  simp [finalStep, selectFinalBranches, selectFinalBranch, fused_recurrent_rwkv6_state_step_slice, ComputeKernel.eraseDType,
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

#axiomsClean final_enabled_stage_shape
end
end VeriTile.Bench.TritonBenchG.FusedRwkv6Kernel.FPStages
