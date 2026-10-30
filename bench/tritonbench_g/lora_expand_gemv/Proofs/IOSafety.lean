import bench.tritonbench_g.lora_expand_gemv.IO
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.IOSafety
open VeriTile.Triton FP.Structural FPValue
open FP.Equational (ReductionPlan)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false
attribute [local irreducible] FPValue.castWord FPValue.mulWord FPValue.addWord FPValue.treeValue

/-- The only prologue access is the signed metadata value for this batch. -/
theorem header_safe (rest : Algebra Word) (plans : ∀ n, ReductionPlan n)
    (g : Geometry) (bounds : RegionName → Nat) (s : State Word)
    (hi : s.pids 1<bounds "I") :
    Safety.traceSafeList (Branches.engine rest .ieeeValue plans) bounds
      (Branches.header .singleHalf .single g.parameters) s := by
  dsimp only [Branches.header,Branches.source,
    VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.source,ComputeKernel.surfaceBody]
  simpa [
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,run,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    State.setReg,Region.cast] using hi

end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.IOSafety
