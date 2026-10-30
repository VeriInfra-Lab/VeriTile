import bench.tritonbench_g.fast_rms_layernorm.Proofs.RuntimeForward
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.NativeSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite Safety
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 10000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Bounds name the actual finite source addresses; arithmetic primitives
never supply a safety or execution oracle. -/
theorem forward (M : Algebra α) (h : Call) (epsilon : α) (s : State α)
    (bounds : RegionName → Nat)
    (x : ∀ j : Fin h.block,j.val<h.cols → s.pids 0*h.cols+j.val<bounds "x")
    (w : ∀ j : Fin h.block,j.val<h.cols → j.val<bounds "w")
    (y : ∀ j : Fin h.block,j.val<h.cols → s.pids 0*h.cols+j.val<bounds "y")
    (r : s.pids 0<bounds "r") :
    TraceSafe (engine M) bounds (Kernels.forwardRuntime h) (RuntimeForward.Source.bind s epsilon) := by
  cases mode : h.gemma
  all_goals
    simp (config := {maxSteps:=500000}) [TraceSafe,Kernels.forwardRuntime,
      Kernels.regularRuntimeSource,Kernels.gemmaRuntimeSource,RuntimeForward.Source.bind,mode,
      ComputeKernel.surfaceBody,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,
      safeCompute_unfold,safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,
      maskActive_unfold,run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,
      store,Option.bind,engine,Algebra.withDefaultPrecision,resolvePrecision,
      ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,Region.cast,
      State.setReg,TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,ofFloat,toFloat]
  all_goals exact ⟨x,w,r,y⟩

theorem backward (M : Algebra α) (h : Call) (s : State α) (bounds : RegionName → Nat)
    (dx : ∀ j : Fin h.block,j.val<h.cols → s.pids 0*h.cols+j.val<bounds "dx")
    (x : ∀ j : Fin h.block,j.val<h.cols → s.pids 0*h.cols+j.val<bounds "x")
    (w : ∀ j : Fin h.block,j.val<h.cols → j.val<bounds "w")
    (r : s.pids 0<bounds "r") :
    TraceSafe (engine M) bounds (Kernels.backward h) s := by
  cases mode : h.gemma
  all_goals
    simp (config := {maxSteps:=500000}) [TraceSafe,Kernels.backward,Kernels.inputGradientSource,mode,
      ComputeKernel.surfaceBody,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,
      safeCompute_unfold,safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,
      maskActive_unfold,run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,
      store,Option.bind,engine,Algebra.withDefaultPrecision,resolvePrecision,
      ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,Region.cast,
      State.setReg,TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,ofFloat,toFloat]
  all_goals exact ⟨dx,x,w,r,dx⟩

end
end VeriTile.Bench.Repairs.FastRmsLayernorm.NativeSafety
