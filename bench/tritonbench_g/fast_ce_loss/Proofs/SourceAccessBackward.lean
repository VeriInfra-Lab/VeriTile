import bench.tritonbench_g.fast_ce_loss.Proofs.KernelBackward
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.FastCeLoss.SourceAccess
open VeriTile.Triton FP.Structural KernelSource KernelModel KernelExecution
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option maxErrors 8
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- The actual in-place backward source loads the saved LSE and active tile;
the ignored-label branch does not load the upstream gradient. -/
theorem backward_safe (M : Algebra α) (bounds : RegionName → Nat) (p : Parameters)
    (initial : State α)
    (labelBound : initial.pids 0 < bounds p.labels.cast)
    (logitsBound : ∀ lane : Fin p.block, initial.pids 1*p.block+lane.val < p.vocab →
      initial.pids 0*p.logitsStride+(initial.pids 1*p.block+lane.val) < bounds p.logits)
    (upstreamBound : label p initial ≠ -100 →
      initial.pids 0*p.upstreamStride < bounds p.upstream)
    (lseBound : initial.pids 0 < bounds p.lse) :
    TraceSafe (engine M) bounds (backward p) initial := by
  classical
  cases hc : p.hasSoftcap <;> cases hs : p.hasScaling
  all_goals
    by_cases ignored : label p initial = -100
    all_goals
      have readIgnored := ignored
      simp only [label,Region.cast] at readIgnored
      simp [TraceSafe,backward,TypedSource.backward,hc,hs,ComputeKernel.surfaceBody,
        Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
        Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
        Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
        run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
        engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
        numeric,natLt,numericLt,numericLe,numericEq,bop,ofFloat,toFloat,Region.cast,
        State.setReg,FP.Structural.store,TileShape.axisDim,Function.comp_def,
        readIgnored,labelBound,lseBound,logitsBound,ignored,upstreamBound]
      exact ⟨labelBound,logitsBound⟩

#axiomsClean backward_safe
end
end VeriTile.Bench.Repairs.FastCeLoss.SourceAccess
