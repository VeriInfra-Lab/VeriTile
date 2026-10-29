import bench.repairs.tritonbench_g.rms_rbe_matmul_launch_guard.Proofs.PublicSourceSafety

namespace VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.PublicIO
open VeriTile.Bench.Repairs.RmsRbeMatmul
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 15000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem stores_safe (a : Operations α) (fp8 : Bool) (p : Parameters)
    (bounds : RegionName → Nat) (s t : State α) (values : TileIndex [16,64] → α)
    (h : TailReady p s t values) (backed : SourceBounds bounds p s) :
    Safety.traceSafeList (engine a) bounds (stores fp8 p) t := by
  cases fp8
  all_goals
    simp [stores,kernel,TypedSource.kernel,RmsMatmulRbe.TypedSource.kernel,arguments,
      ComputeKernel.surfaceBody,Safety.traceSafeList_cons,Safety.stepSafe_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
      step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
      numeric,bop,natLt,numericLt,integralFloorDiv,integralMod,Region.cast,ofFloat,toFloat,
      State.setReg,engine,Algebra.withDefaultPrecision,resolvePrecision,
      h.batch,h.tileM,h.tileN,h.product,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals intro i j hi hj
  all_goals simpa [outputOffset,row,col,Nat.mul_one] using
    backed.output (i,j,PUnit.unit) (by simpa [active,row,col] using And.intro hi hj)

#axiomsClean stores_safe
end
end VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.PublicIO
