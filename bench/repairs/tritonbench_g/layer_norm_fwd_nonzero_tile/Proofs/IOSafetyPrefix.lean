import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Source
import bench.tritonbench_g.layer_norm_fwd.Proofs.FlatDomain
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.IOSafety
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwd
open TypedSource TypedSupport
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem prefix_safe {α : Type} [Inhabited α] (A : Algebra α)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (s : State α)
    (bounds : RegionBounds) (access : FlatDomain.RowBounds f r stride M N (s.pids 0) bounds) :
    Safety.traceSafeList (engine A) bounds (stagePrefix f r stride M N eps p block hb) s := by
  rcases f with ⟨rms,res,resout,bias,drop,mask,scale,x1,w1,b1⟩
  cases res <;> cases resout <;> cases x1 <;> cases w1
  all_goals simp (config := {maxSteps:=500000}) [stagePrefix,repaired,original,ComputeKernel.surfaceBody,
    Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
    Safety.maskActive_unfold,Safety.precisionContext,
    run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,integralMod,bop,natLt,
    numericLt,ofFloat,toFloat,Region.cast,Random.i32ArangeSize,
    State.setReg,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals intro j on; exact access.x j.val on

#axiomsClean prefix_safe
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.IOSafety
