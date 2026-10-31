import bench.tritonbench_g.lora_expand_gemv.Proofs.IOSafety

namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.IOSafety
open VeriTile.Triton FP.Structural FPValue
open FP.Equational (ReductionPlan)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false
attribute [local irreducible] FPValue.castWord FPValue.mulWord FPValue.addWord FPValue.treeValue
attribute [local simp] VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.inputFormat
  VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.weightFormat
  VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.effectiveFormat

theorem setup_safe (rest : Algebra Word) (plans : ∀ n, ReductionPlan n)
    (g : Geometry) (limits : RegionName → Nat) (s : State Word)
    (hx : ∀ k : Fin g.block,g.even=Bool.true ∨ k.val<g.K →
      xoff g (s.pids 1) k.val<limits "X") :
    Safety.traceSafeList (Branches.engine rest .ieeeValue plans) limits
      (Branches.setup .singleHalf .single g.parameters)
      (Branches.headerExecution rest .ieeeValue plans .singleHalf .single g.parameters s).val := by
  dsimp only [Branches.setup,Branches.active,Branches.castBranch,Branches.source,
    VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.source,ComputeKernel.surfaceBody,VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.fullSource,
    Geometry.parameters]
  cases he : g.even <;>
    simp (config := {maxSteps := 500000}) [Branches.setup,Branches.active,Branches.castBranch,
      Branches.source,VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.source,ComputeKernel.surfaceBody,VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.fullSource,
      Geometry.parameters,he,Branches.headerExecution,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      step,run,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      State.setReg,Branches.engine,FPValue.backend,FPValue.formatOf,
      numeric,natLt,numericEq,bop,ComputeDType.eraseDType,ofFloat,toFloat,
      FloatDType.numericDType,FloatDType.toTileDType,
      Broadcast.leftIndex,Broadcast.rightIndex,Region.cast]
  all_goals try simp only [Safety.traceSafeList, Safety.stepSafe]
  · intro k hk
    exact hx k (Or.inr hk)
  · intro k
    exact hx k (Or.inl he)

end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.IOSafety
