import bench.tritonbench_g.quantize_global.Proofs.Source
import VeriTile.Triton.Float.IntegerStageSafety
namespace QuantizeGlobalRepair.KernelSafety
open VeriTile.Triton FP.Structural QuantizeIntegerStage
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- All accesses of the actual div.rn/llrint64 source, including the
unconditional scalar maximum load and the masked INT8 store. -/
theorem source_safe {α : Type} [Inhabited α] (rest : Algebra α) (a : Scalar α)
    (s : State α) (X Maximum Out : RegionName) (N base B : Nat)
    (bounds : RegionName → Nat)
    (hx : ∀ i, i < N → i < bounds X)
    (hm : 0 < bounds Maximum)
    (ho : ∀ i, i < N → i < bounds Out) :
    QuantizeIntegerStage.TraceSafe (backend rest a) (some a.round64) bounds
      (kernel X Maximum Out N base B) s := by
  simp +contextual (config := {maxSteps := 1000000})
    [QuantizeIntegerStage.TraceSafe, kernel, source, ComputeKernel.surfaceBody,
     QuantizeIntegerStage.Safety.traceSafeList_unfold,
     QuantizeIntegerStage.Safety.stepSafe_unfold,
     QuantizeIntegerStage.Safety.safeExpr_unfold,
     FP.Structural.Safety.safeExpr_unfold, FP.Structural.Safety.safeCompute_unfold,
     FP.Structural.Safety.safeOp_unfold, FP.Structural.Safety.safeAddress_unfold,
     FP.Structural.Safety.safeMask_unfold, FP.Structural.Safety.addressSafe_unfold,
     FP.Structural.Safety.maskActive_unfold,
     QuantizeIntegerStage.step, QuantizeIntegerStage.evalExpr, QuantizeIntegerStage.store,
     storedInteger, byteOutput, outputRegister,
     FP.Structural.step, stepAlg, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
     FP.Structural.store, backend, State.setReg, TileShape.axisDim,
     ofFloat, toFloat, Region.cast, ComputeDType.eraseDType,
     bop, numeric, numericLt, numericEq, natLt, Option.bind, hx, hm, ho]
  constructor
  · intro j active
    simpa only [if_pos active] using hx _ active
  · intro j active
    simpa only [if_pos active] using ho _ active

#axiomsClean source_safe
end
end QuantizeGlobalRepair.KernelSafety
