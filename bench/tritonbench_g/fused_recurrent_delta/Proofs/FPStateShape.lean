import bench.tritonbench_g.fused_recurrent_delta.Proofs.FPState
import bench.tritonbench_g.fused_recurrent_hgrn.Proofs.FPShape
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentDelta.FPStages
open VeriTile.Triton FP.Structural
open FusedRecurrentHgrn.FPStages (erase_real_cast)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 15000
set_option linter.unusedSimpArgs false

/-- Select only syntactically constant boolean branches. This does not
simplify floating arithmetic, masks or memory operations. -/
def selectBranch (st : ComputeStmt) : List ComputeStmt :=
  match st with
  | .ifThenElse (.constBool b) yes no => if b then yes else no
  | _ => [st]

def selectBranches (body : List ComputeStmt) : List ComputeStmt := body.flatMap selectBranch

theorem selectBranch_exec {α : Type} [Inhabited α] (M : Algebra α) (st : ComputeStmt) (s : State α) :
    run M (selectBranch st) s = step M st s := by
  unfold selectBranch
  split
  · rename_i b yes no
    cases b <;> simp [step, evalOp_unfold, Option.bind]
  · simp [run]


theorem selectBranches_exec {α : Type} [Inhabited α] (M : Algebra α) (body : List ComputeStmt) (s : State α) :
    run M (selectBranches body) s = run M body s := by
  induction body generalizing s with
  | nil => rfl
  | cons st rest ih =>
    simp only [selectBranches, List.flatMap_cons, run_append, selectBranch_exec, run]
    simp only [selectBranches] at ih
    cases h : step M st s <;> simp [h, ih]

theorem state_headwise_shape (HPrev k v beta HOut : RegionName) (t s_qk_h s_vo_h T K V BK BV : Nat) :
    (ComputeKernel.mk [HPrev,k,v,beta] [HOut]
      (selectBranches (stateStep HPrev k v beta HOut Bool.true t s_qk_h s_vo_h T K V BK BV).surfaceBody)).eraseDType.surfaceBody =
      (fused_recurrent_delta_state_step_slice_headwise HPrev k v beta HOut t s_qk_h s_vo_h T K V BK BV).eraseDType.surfaceBody := by
  simp [stateStep, fused_recurrent_delta_state_step_slice_headwise, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    ComputeKernel.fromKernelBody, ComputeKernel.surfaceBody, ComputeDType.eraseDType,
    Except.bind, bind, pure, erase_real_cast, selectBranches, selectBranch]
  repeat' constructor
  all_goals conv_lhs => rw [Op.eraseDType.eq_def]
  all_goals simp [Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, VeriTile.Triton.eraseDType]

theorem state_scalar_shape (HPrev k v beta HOut : RegionName) (t s_qk_h s_vo_h T K V BK BV : Nat) :
    (ComputeKernel.mk [HPrev,k,v,beta] [HOut]
      (selectBranches (stateStep HPrev k v beta HOut Bool.false t s_qk_h s_vo_h T K V BK BV).surfaceBody)).eraseDType.surfaceBody =
      (fused_recurrent_delta_state_step_slice_scalarbeta HPrev k v beta HOut t s_qk_h s_vo_h T K V BK BV).eraseDType.surfaceBody := by
  simp [stateStep, fused_recurrent_delta_state_step_slice_scalarbeta, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    ComputeKernel.fromKernelBody, ComputeKernel.surfaceBody, ComputeDType.eraseDType,
    Except.bind, bind, pure, erase_real_cast, selectBranches, selectBranch]
  repeat' constructor
  all_goals conv_lhs => rw [Op.eraseDType.eq_def]
  all_goals simp [Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, VeriTile.Triton.eraseDType]

#axiomsClean selectBranch_exec
#axiomsClean selectBranches_exec
#axiomsClean state_headwise_shape
#axiomsClean state_scalar_shape
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentDelta.FPStages
