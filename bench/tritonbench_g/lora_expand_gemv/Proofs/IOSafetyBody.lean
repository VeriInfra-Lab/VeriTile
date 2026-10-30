import bench.tritonbench_g.lora_expand_gemv.Proofs.IOSafety

namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.IOSafety
open VeriTile.Triton FP.Structural FPValue
open FP.Equational (ReductionPlan)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false
attribute [local irreducible] FPValue.castWord FPValue.mulWord FPValue.addWord FPValue.treeValue
attribute [local simp] VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.inputFormat
  VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.weightFormat
  VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.effectiveFormat
  VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.promote
  VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.productFormat
  VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.resultFormat

theorem body_safe (rest : Algebra Word) (plans : ∀ n, ReductionPlan n)
    (g : Geometry) (limits : RegionName → Nat) (initial current : State Word)
    (h : Branches.Invariant .ieeeValue .singleHalf g.parameters initial current)
    (hw : ∀ (m : Fin 256) (k : Fin g.block),m.val<Branches.length g.parameters → k.val<g.K →
      Branches.bOffset g.parameters initial m.val k.val<limits "W")
    (hy : ∀ m : Fin 256,m.val<Branches.length g.parameters →
      Branches.outputOffset g.parameters initial m.val<limits "Y") :
    Safety.traceSafeList (Branches.engine rest .ieeeValue plans) limits
      (Branches.body .singleHalf .single g.parameters)
      (current.setReg "n" .nat [] (fun _ => 0)) := by
  have hoffk := h.offk
  have hoffn := h.offn
  have hbp := h.bp
  have hcp := h.cp
  have hsize := h.size
  dsimp only [Geometry.parameters] at hoffk hoffn hbp hcp hsize
  have hav := h.av
  simp [Branches.dtype,Geometry.parameters,FloatDType.toTileDType,ofFloat] at hav
  dsimp only [Branches.body,Branches.castBranch,Branches.active,Branches.source,
    VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.source,ComputeKernel.surfaceBody,
    Geometry.parameters]
  cases ha : g.add <;>
    simp (config := {maxSteps := 800000}) [Branches.body,Branches.castBranch,Branches.active,
      Branches.source,VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.source,ComputeKernel.surfaceBody,
      Geometry.parameters,ha,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      step,run,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,store,
      hoffk,hoffn,hbp,hcp,hsize,hav,
      State.setReg,Branches.engine,FPValue.backend,FPValue.formatOf,FPValue.typedSum,
      numeric,natLt,numericEq,bop,ComputeDType.eraseDType,ofFloat,toFloat,
      FloatDType.numericDType,FloatDType.toTileDType,
      Broadcast.leftIndex,Broadcast.rightIndex,Region.cast]
  all_goals simpa only [Branches.bOffset,Branches.outputOffset,Geometry.parameters,ha]
    using And.intro hw hy

end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.IOSafety
