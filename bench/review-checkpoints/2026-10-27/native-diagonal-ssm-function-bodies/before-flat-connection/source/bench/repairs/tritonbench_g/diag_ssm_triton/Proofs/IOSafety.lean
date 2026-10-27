import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.ReadableSource
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.IOForwardSafety
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.IOBackwardSafetyBase

namespace VeriTile.Bench.Repairs.DiagSsmTriton.IOSafety
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem backward_trace (p : BackwardLayout) (length : Nat)
    (sep : RegionCorrectness.BackwardSeparation p) (u : BlockState) (bounds : RegionBounds)
    (hb : BackwardIO.Bounds p u length bounds) :
    (Kernels.backwardTyped p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).eraseDType.toAlgKernel.TraceSafe bounds u := by
  apply (SourceBackward.backward_trace p length u bounds).mpr
  apply (SourceReferenceTrace.backward p length bounds u).mpr
  by_cases hz : length = 0
  · subst length
    have z := ZeroSafety.backward_safe bounds u p.s p.l p.y p.gs p.gx p.gl p.gy p.batch p.dim p.B
      ⟨hb.transition, hb.gradInitial, hb.gradTransition⟩
    simp only [Kernel.TraceSafe, BackwardProjection.real_source_body, List.append_assoc,
      TraceRewrite.append_iff] at z
    simp only [Kernel.TraceSafeR, RealReference.diag_ssm_backward_kernel_toAlg_body,
      RealReference.diagSsmBackwardProjectedBody, List.append_assoc, appendR_iff]
    refine ⟨(backward_prefix bounds p.l p.batch p.dim p.B u).mp z.1, ?_⟩
    intro v he
    have heOld : stepStmts (BackwardProjection.prelude p.l p.batch p.dim p.B) u = some v := by
      simpa only [BackwardProjection.prelude_steps] using he
    have tail := z.2 v heOld
    have hzLoop : stepStmts [.forLoop "i" 0
        (BackwardProjection.loopBody p.s p.y p.gx p.gy 0 p.batch p.dim p.B)] v = some v := by
      simp only [stepStmts, stepForLoopAux.forLoop_unfold,
        stepForLoopAux.step_ge (Nat.le_refl 0)]
    have post := (backward_post bounds p.gs p.gl p.B v).mp (tail.2 v hzLoop)
    simp only [Stmt.TraceSafeListR, Stmt.TraceSafeR, Stmt.forRangeTraceSafeR,
      Nat.one_ne_zero, Nat.lt_irrefl, ↓reduceIte, true_and, stepStmtR_triv,
      stepStmts, stepForRangeAux.forRange_unfold, stepForRangeAux.step_ge Nat.one_ne_zero (Nat.le_refl 0),
      Option.some.injEq, forall_eq']
    simpa only [true_and] using post
  · have hp : 0 < length := by omega
    exact RealReference.backward_traceSafeR .triv bounds p.s p.l p.y p.gs p.gx p.gl p.gy
      length p.batch p.dim p.B hp sep.initial sep.output sep.gradient sep.transition u
      (fun t => hb.gradOutput t.val t.isLt)
      (fun t j hj => hb.previous (t.val-1) (by have := t.isLt; omega) j hj.2)
      (fun _ j hj => hb.initial hp j hj.2) (fun _ => hb.transition)
      (fun t => hb.gradInput t.val t.isLt)
      (fun _ j hj => hb.gradInitial j hj.2) (fun _ j hj => hb.gradTransition j hj.2)

#axiomsClean forward_trace
#axiomsClean backward_trace

end
end VeriTile.Bench.Repairs.DiagSsmTriton.IOSafety
