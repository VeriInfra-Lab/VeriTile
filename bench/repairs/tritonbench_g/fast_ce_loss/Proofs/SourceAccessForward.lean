import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.KernelForward
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.FastCeLoss.SourceAccess
open VeriTile.Triton FP.Structural KernelSource KernelModel KernelExecution
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option maxErrors 8
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Actual forward accesses, including the independently backed scalar gather
for every admitted nonignored native label. No vocabulary upper bound is added. -/
theorem forward_safe (M : Algebra α) (bounds : RegionName → Nat) (p : Parameters) (initial : State α)
    (positive : 0 < p.block)
    (labelBound : initial.pids 0 < bounds p.labels.cast)
    (logitsBound : ∀ lane : Fin p.block, lane.val < p.vocab →
      initial.pids 0*p.logitsStride+lane.val < bounds p.logits)
    (gatherBound : label p initial ≠ -100 →
      initial.pids 0*p.logitsStride+(label p initial).toNat-(-(label p initial)).toNat < bounds p.logits)
    (lossBound : initial.pids 0 < bounds p.loss)
    (lseBound : initial.pids 0 < bounds p.lse) :
    TraceSafe (engine M) bounds (forward p) initial := by
  classical
  cases hc : p.hasSoftcap <;> cases hs : p.hasScaling
  all_goals
    by_cases ignored : label p initial = -100
    all_goals
      have readIgnored := ignored
      simp only [label,Region.cast] at readIgnored
      simp [TraceSafe,forward,TypedSource.forward,hc,hs,ComputeKernel.surfaceBody,
        Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
        Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
        Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
        run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
        engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
        numeric,natLt,numericLt,numericLe,numericEq,bop,ofFloat,toFloat,Region.cast,
        State.setReg,FP.Structural.store,TileShape.axisDim,positive,Function.comp_def,
        readIgnored,labelBound,lossBound,lseBound,logitsBound,gatherBound,ignored]
      first
      | exact ⟨labelBound,logitsBound⟩
      | exact ⟨labelBound,logitsBound,gatherBound ignored⟩

#axiomsClean forward_safe
end
end VeriTile.Bench.Repairs.FastCeLoss.SourceAccess
