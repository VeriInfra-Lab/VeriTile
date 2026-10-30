import bench.repairs.tritonbench_g.rbe_triton_transform_checked_launch.Proofs.IOCorrectness
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch.Pair
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RbeTritonTransform
noncomputable section
set_option maxHeartbeats 24000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

/-- Reusing the masks has identical full state semantics, for arbitrary payloads. -/
theorem kernel_run {β : Type} [Inhabited β] (M : Algebra β)
    (p : Source.Parameters) (s : State β) :
    FP.Structural.exec M (maskReuse p) s = FP.Structural.exec M (source p) s := by
  simp [source, typed, maskReuse, typedMaskReuse, FP.Structural.exec,
    ComputeKernel.surfaceBody, FP.Structural.run, FP.Structural.step,
    FP.Structural.evalExpr, FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold,
    FP.Structural.evalOffsets, FP.Structural.store, FP.Structural.numeric,
    FP.Structural.bop, FP.Structural.natLt, FP.Structural.numericLt, FP.Structural.numericEq,
    ComputeDType.eraseDType, FP.Structural.ofFloat, FP.Structural.toFloat,
    FloatDType.numericDType, FloatDType.toTileDType, Region.cast, Option.bind,
    integralFloorDiv, integralMod]

/-- The original and candidate perform exactly the same masked accesses. -/
theorem kernel_safe {β : Type} [Inhabited β] (M : Algebra β)
    (p : Source.Parameters) (bounds : RegionName → Nat) (s : State β) :
    TraceSafe M bounds (maskReuse p) s ↔ TraceSafe M bounds (source p) s := by
  simp (maxSteps := 1800000) [TraceSafe, source, typed, maskReuse, typedMaskReuse,
    ComputeKernel.surfaceBody, Safety.traceSafeList_unfold, Safety.stepSafe_unfold,
    Safety.safeExpr_unfold, Safety.safeCompute_unfold, Safety.safeOp_unfold,
    Safety.safeAddress_unfold, Safety.safeMask_unfold, Safety.addressSafe_unfold,
    Safety.maskActive_unfold, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets,
    store, Option.bind, numeric, bop, natLt, numericLt, numericEq, ofFloat, toFloat,
    Region.cast, State.setReg, ComputeDType.eraseDType,
    FloatDType.numericDType, FloatDType.toTileDType, integralFloorDiv, integralMod]

theorem invocation_run {β : Type} [Inhabited β] (ops : Operations β) (r : Request)
    (X Y : RegionName) (q : Runtime.Program) (s : State β) :
    (invocation ops r X Y q).run s = (RbeTritonTransformCheckedLaunch.invocation ops r X Y q).run s :=
  kernel_run ops.backend _ _

theorem invocation_safe {β : Type} [Inhabited β] (ops : Operations β) (r : Request)
    (X Y : RegionName) (q : Runtime.Program) (bounds : RegionName → Nat) (s : State β) :
    (invocation ops r X Y q).Safe bounds s ↔
      (RbeTritonTransformCheckedLaunch.invocation ops r X Y q).Safe bounds s :=
  kernel_safe ops.backend _ bounds _

theorem schedule_run {β : Type} [Inhabited β] (ops : Operations β) (r : Request)
    (X Y : RegionName) (qs : List Runtime.Program) (s : State β) :
    runSchedule (qs.map (invocation ops r X Y)) s =
      runSchedule (qs.map (RbeTritonTransformCheckedLaunch.invocation ops r X Y)) s := by
  induction qs generalizing s with
  | nil => rfl
  | cons q qs ih => simp only [List.map_cons, runSchedule, invocation_run, ih]

theorem schedule_safe {β : Type} [Inhabited β] (ops : Operations β) (r : Request)
    (X Y : RegionName) (qs : List Runtime.Program) (bounds : RegionName → Nat) (s : State β) :
    ScheduleSafe bounds (qs.map (invocation ops r X Y)) s ↔
      ScheduleSafe bounds (qs.map (RbeTritonTransformCheckedLaunch.invocation ops r X Y)) s := by
  induction qs generalizing s with
  | nil => rfl
  | cons q qs ih => simp only [List.map_cons, ScheduleSafe, invocation_safe, invocation_run, ih]

/-- Existing full-grid source correctness supplies success and frame; the pair
connection preserves every actual launch and both source safety conclusions. -/
theorem equivalent {α β : Type} [Inhabited α] [Inhabited β]
    (ops : Operations β) (r : Request) (X Y : RegionName) (different : X≠Y)
    (qs : List Runtime.Program) (all : qs.Perm (Runtime.programs r))
    (encode : α → β) (observe : β → α) :
    (rotationIO ops r X Y qs encode observe).Equivalent (variant ops r X Y qs encode observe) := by
  intro values domain bounds bound initial binding
  obtain ⟨safe,final,ran,outputs,frame⟩ :=
    IOCorrectness.implements ops r X Y different qs all encode observe
      values domain bounds bound initial binding
  refine ⟨safe,?_,final,final,ran,?_,?_,frame,frame⟩
  · exact (schedule_safe ops r X Y qs bounds initial).mpr safe
  · exact (schedule_run ops r X Y qs initial).trans ran
  · intro o j active
    rfl

#axiomsClean kernel_run
#axiomsClean kernel_safe
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch.Pair
