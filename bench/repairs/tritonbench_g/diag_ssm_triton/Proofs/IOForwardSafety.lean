import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.ReadableSource
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.IOSafetyBase

namespace VeriTile.Bench.Repairs.DiagSsmTriton.IOSafety
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem forward_trace (p : ForwardLayout) (length : Nat) (sep : p.x ≠ p.y)
    (u : BlockState) (bounds : RegionBounds)
    (hb : ForwardIO.Bounds p u length bounds) :
    (Kernels.forwardTyped p.s p.x p.l p.y length p.batch p.dim p.B).eraseDType.toAlgKernel.TraceSafe bounds u := by
  by_cases hz : length = 0
  · subst length
    simpa only [Kernel.TraceSafe,Kernels.ReadableSource.forward_preserved,Erasure.forward_body] using
      ZeroSafety.forward_safe bounds u p.s p.x p.l p.y p.batch p.dim p.B ⟨hb.initial,hb.transition⟩
  · have hp : 0 < length := by omega
    have old := RealReference.forward_traceSafeR .triv bounds p.s p.x p.l p.y length p.batch p.dim p.B
      hp sep u (fun _ => hb.initial) (fun _ => hb.transition)
      (fun t => hb.input t.val t.isLt) (fun t => hb.output t.val t.isLt)
    simp only [Kernel.TraceSafeR,RealReference.diag_ssm_forward_kernel_toAlg_body,
      RealReference.diagSsmForwardProjectedBody,appendR_iff] at old
    simp only [Kernel.TraceSafe,Kernels.ReadableSource.forward_preserved,Erasure.forward_body,
      ForwardProjection.real_source_body,TraceRewrite.append_iff]
    refine ⟨(forward_prefix bounds p.s p.l p.batch p.dim p.B u).mpr old.1,?_⟩
    intro v he
    rw [ForwardProjection.prelude_steps] at he
    have tail := old.2 v he
    simp only [Stmt.TraceSafeListR,Stmt.TraceSafeR,stepStmtR_triv] at tail
    simp only [Stmt.TraceSafeList,Stmt.TraceSafe]
    refine ⟨?_,?_⟩
    · exact (loop_iff bounds "t" 0 length _ _
        (ForwardProjection.loop_steps p.x p.y p.batch p.dim p.B)
        (forward_body bounds p.x p.y p.batch p.dim p.B) v).mpr tail.1
    · cases stepStmt (.forLoop "t" length (ForwardProjection.loopBody p.x p.y p.batch p.dim p.B)) v <;>
        simp [Stmt.TraceSafeList]


end
end VeriTile.Bench.Repairs.DiagSsmTriton.IOSafety
