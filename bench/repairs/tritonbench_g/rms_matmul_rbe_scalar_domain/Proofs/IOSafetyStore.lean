import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.IOSafetyBasics

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOSafety
open VeriTile.Triton FP.Structural Safety
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem storing_safe (a : Arithmetic α) (outputFP8 : Bool) (p : Arguments) (s : State α)
    (pb pm pn : Nat) (accumulator : Values α .real [16,64])
    (hb : s.regs .nat [] "pid_batch"=some (fun _=>pb))
    (hm : s.regs .nat [] "pid_m"=some (fun _=>pm))
    (hn : s.regs .nat [] "pid_n"=some (fun _=>pn))
    (ha : s.regs .real [16,64] "accumulator"=some accumulator)
    (capacity : RegionName → Nat)
    (bound : ∀ i : TileIndex [16,64],outputActive p pm pn i → outputOffset p pb pm pn i<capacity p.OUT) :
    traceSafeList a.model capacity (storing outputFP8 p) s := by
  cases outputFP8
  all_goals
    simp [storing,epilogue,source,TypedSource.kernel,TypedSource.halfHalf,TypedSource.halfFP8,TypedSource.fp8Half,TypedSource.fp8FP8,ComputeKernel.surfaceBody,
      traceSafeList_unfold,stepSafe_unfold,safeAddress_unfold,safeMask_unfold,
      safeExpr_unfold,safeCompute_unfold,safeOp_unfold,addressSafe_unfold,maskActive_unfold,
      FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,
      numeric,bop,natLt,numericLt,ComputeDType.eraseDType,ofFloat,toFloat,
      Region.cast,Option.bind,integralFloorDiv,integralMod,Arithmetic.model,hb,hm,hn,ha,
      store,Broadcast.leftIndex,Broadcast.rightIndex]
    intro m n hm hn
    exact bound (m,n,PUnit.unit) ⟨hm,hn⟩

#axiomsClean storing_safe
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOSafety
