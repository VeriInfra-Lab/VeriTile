import bench.tritonbench_g.rbe_triton_transform.Proofs.Runtime
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch.AccessSafety
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RbeTritonTransform
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- Both masked components of the selected quotation access exactly their
published input/output windows. The floating backend does not affect addresses. -/
theorem source {α : Type} [Inhabited α] (M : Algebra α) (p : Source.Parameters)
    (bounds : RegionName → Nat) (s : State α)
    (input : ∀ imag i, Source.active p s imag i →
      Source.inputOffset p s imag i < bounds p.input)
    (output : ∀ imag i, Source.active p s imag i →
      Source.outputOffset p s imag i < bounds p.output) :
    TraceSafe M bounds (Selected.source p) s := by
  simp (maxSteps := 1200000) [TraceSafe,Selected.source,Selected.typed,VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch.typed,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
    Safety.maskActive_unfold,Safety.precisionContext,
    FP.Structural.run,FP.Structural.step,FP.Structural.evalExpr,
    FP.Structural.evalComputeOp,FP.Structural.evalOp_unfold,
    FP.Structural.evalOffsets,FP.Structural.store,FP.Structural.numeric,
    FP.Structural.bop,FP.Structural.natLt,FP.Structural.numericEq,ComputeDType.eraseDType,
    FP.Structural.ofFloat,FP.Structural.toFloat,FloatDType.numericDType,FloatDType.toTileDType,
    Region.cast,Option.bind,integralFloorDiv,integralMod,
    Source.active,Source.inputOffset,Source.outputOffset,Source.row,Source.column,Source.parts] at input output ⊢
  all_goals tauto

#axiomsClean source
end
end VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch.AccessSafety
