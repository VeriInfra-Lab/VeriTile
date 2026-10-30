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
  by_cases hz : length = 0
  · subst length
    simpa only [Kernel.TraceSafe,Kernels.ReadableSource.backward_preserved,Erasure.backward_body] using
      ZeroSafety.backward_safe bounds u p.s p.l p.y p.gs p.gx p.gl p.gy p.batch p.dim p.B
        ⟨hb.transition,hb.gradInitial,hb.gradTransition⟩
  · have hp : 0 < length := by omega
    have old := RealReference.backward_traceSafeR .triv bounds p.s p.l p.y p.gs p.gx p.gl p.gy
      length p.batch p.dim p.B hp sep.initial sep.output sep.gradient sep.transition u
      (fun t => hb.gradOutput t.val t.isLt)
      (fun t j hj => hb.previous (t.val-1) (by have := t.isLt; omega) j hj.2)
      (fun _ j hj => hb.initial hp j hj.2) (fun _ => hb.transition)
      (fun t => hb.gradInput t.val t.isLt)
      (fun _ j hj => hb.gradInitial j hj.2) (fun _ j hj => hb.gradTransition j hj.2)
    simp only [Kernel.TraceSafeR,RealReference.diag_ssm_backward_kernel_toAlg_body,
      RealReference.diagSsmBackwardProjectedBody,List.append_assoc,appendR_iff] at old
    simp only [Kernel.TraceSafe,Kernels.ReadableSource.backward_preserved,Erasure.backward_body,
      BackwardProjection.real_source_body,List.append_assoc,TraceRewrite.append_iff]
    refine ⟨(backward_prefix bounds p.l p.batch p.dim p.B u).mpr old.1,?_⟩
    intro v he
    rw [BackwardProjection.prelude_steps] at he
    have tail := old.2 v he
    simp only [List.singleton_append,Stmt.TraceSafeListR,Stmt.TraceSafeR,stepStmtR_triv] at tail
    have hl := (loop_range_iff bounds "i" 0 length _ _
      (BackwardProjection.loop_steps p.s p.y p.gx p.gy length p.batch p.dim p.B)
      (backward_body bounds p.s p.y p.gx p.gy length p.batch p.dim p.B) v).mpr tail.1.1
    refine ⟨Stmt.TraceSafeList.cons_intro (by simpa only [Stmt.TraceSafe] using hl) (fun _ _ => Stmt.TraceSafeList.nil_intro),?_⟩
    have heq : stepStmt (.forLoop "i" length (BackwardProjection.loopBody p.s p.y p.gx p.gy length p.batch p.dim p.B)) v =
        stepStmt (.forRange "i" 0 length 1 (RealReference.diagSsmBackwardLoopBody p.s p.y p.gx p.gy length p.batch p.dim p.B)) v := by
      simp only [stepStmt,stepForLoopAux.forLoop_unfold,stepForRangeAux.forRange_unfold]
      exact BackwardProjection.loop_range "i" 0 length _ _
        (BackwardProjection.loop_steps p.s p.y p.gx p.gy length p.batch p.dim p.B) v
    intro w hw
    apply (backward_post bounds p.gs p.gl p.B w).mpr
    apply tail.2 w
    simpa only [stepStmts,heq] using hw

#axiomsClean forward_trace
#axiomsClean backward_trace

end
end VeriTile.Bench.Repairs.DiagSsmTriton.IOSafety
