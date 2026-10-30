import bench.repairs.tritonbench_g.cross_entropy2.Proofs.KernelBackward
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.CrossEntropy2.KernelSafety
open VeriTile.Triton FP.Structural KernelNative
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- The exact forward prefix, through the first scalar LSE store. -/
theorem forward_prefix (M : Algebra α) (p : Parameters) (initial : State α)
    (positive : 0<p.block) (values : RuntimeArgs.Values α)
    (abi : RuntimeArgs.Bound p initial values) (bounds : RegionName → Nat)
    (label : p.labelsBase+initial.pids 0<bounds (Region.cast p.labels))
    (logits : ∀ lane : Fin p.block, initial.pids 1*p.block+lane.val<p.cols →
      p.logitsBase+initial.pids 0*p.logitsStride+(initial.pids 1*p.block+lane.val)<bounds p.logits)
    (lse : p.lseBase+initial.pids 1*p.rows+initial.pids 0<bounds p.lse) :
    Safety.traceSafeList (engine M) bounds (front p) initial := by
  change p.labelsBase+initial.pids 0<bounds {name := p.labels.name} at label
  cases hs : p.hasSmoothing
  all_goals
    simp [front,forwardSyntax,hs,ComputeKernel.surfaceBody,
      Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
      Safety.safeOp,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,
      Safety.addressSafe,Safety.maskActive,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,natLt,numericLt,numericLe,bop,ofFloat,toFloat,Region.cast,
      State.setReg,FP.Structural.store,TileShape.axisDim,positive,Function.comp_def,
      abi.logitScale,label,lse]
    exact logits

/-- Backward guards the upstream load on the actual signed label and masks
both logit loads and gradient stores on the original column index. -/
theorem backward (M : Algebra α) (p : Parameters) (initial : State α)
    (values : RuntimeArgs.Values α) (abi : RuntimeArgs.Bound p initial values)
    (bounds : RegionName → Nat)
    (label : p.labelsBase+initial.pids 0<bounds (Region.cast p.labels))
    (upstream : target p initial≠p.ignored →
      p.upstreamBase+initial.pids 0*p.upstreamStride<bounds p.upstream)
    (logits : ∀ lane : Fin p.block, initial.pids 1*p.block+lane.val<p.cols →
      p.logitsBase+initial.pids 0*p.logitsStride+(initial.pids 1*p.block+lane.val)<bounds p.logits)
    (lse : p.lseBase+initial.pids 0<bounds p.lse)
    (gradient : ∀ lane : Fin p.block, initial.pids 1*p.block+lane.val<p.cols →
      p.gradientBase+initial.pids 0*p.gradientStride+(initial.pids 1*p.block+lane.val)<bounds p.gradient) :
    TraceSafe (engine M) bounds (backwardSyntax p) initial := by
  change p.labelsBase+initial.pids 0<bounds {name := p.labels.name} at label
  cases hs : p.hasSmoothing
  all_goals by_cases owner : ownsTarget p initial
  all_goals by_cases ignored : target p initial=p.ignored
  all_goals
    have upstreamBound := upstream
    simp only [target,Region.cast] at upstreamBound
    simp only [ownsTarget,target,Region.cast] at owner
    simp only [target,Region.cast] at ignored
    simp [TraceSafe,backwardSyntax,hs,ComputeKernel.surfaceBody,
      Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
      Safety.safeOp,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,
      Safety.addressSafe,Safety.maskActive,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,numericEq,natLt,numericLt,numericLe,bop,ofFloat,toFloat,Region.cast,
      State.setReg,FP.Structural.store,TileShape.axisDim,Function.comp_def,
      abi.logitScale,abi.smoothing,abi.squareScale,abi.totalClasses,
      owner,ignored,label,lse,upstreamBound]
    exact ⟨logits,gradient⟩

#axiomsClean forward_prefix
#axiomsClean backward
end
end VeriTile.Bench.Repairs.CrossEntropy2.KernelSafety
