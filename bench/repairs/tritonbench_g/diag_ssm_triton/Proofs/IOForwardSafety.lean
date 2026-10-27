import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.ReadableSource
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceReferenceTrace
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.IOSafetyBase

namespace VeriTile.Bench.Repairs.DiagSsmTriton.IOSafety
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem forward_trace_of_positive (p : ForwardLayout) (length : Nat) (sep : 0 < length → p.x ≠ p.y)
    (u : BlockState) (bounds : RegionBounds)
    (hb : ForwardIO.Bounds p u length bounds) :
    (Kernels.forwardTyped p.s p.x p.l p.y length p.batch p.dim p.B).eraseDType.toAlgKernel.TraceSafe bounds u := by
  apply (SourceReal.forward_trace p length u bounds).mpr
  apply (SourceReferenceTrace.forward p length bounds u).mpr
  by_cases hz : length = 0
  · subst length
    have z := ZeroSafety.forward_safe bounds u p.s p.x p.l p.y p.batch p.dim p.B
      ⟨hb.initial, hb.transition⟩
    simp only [Kernel.TraceSafe, ForwardProjection.real_source_body, TraceRewrite.append_iff] at z
    simp only [Kernel.TraceSafeR, RealReference.diag_ssm_forward_kernel_toAlg_body,
      RealReference.diagSsmForwardProjectedBody, appendR_iff]
    refine ⟨(forward_prefix bounds p.s p.l p.batch p.dim p.B u).mp z.1, ?_⟩
    intro v _
    simp only [Stmt.TraceSafeListR, Stmt.TraceSafeR, Stmt.forLoopTraceSafeR,
      Nat.lt_irrefl, ↓reduceIte, true_and, stepStmtR_triv,
      stepForLoopAux.forLoop_unfold, stepForLoopAux.step_ge (Nat.le_refl 0)]
  · have hp : 0 < length := by omega
    exact RealReference.forward_traceSafeR .triv bounds p.s p.x p.l p.y length p.batch p.dim p.B
      hp (sep hp) u (fun _ => hb.initial) (fun _ => hb.transition)
      (fun t => hb.input t.val t.isLt) (fun t => hb.output t.val t.isLt)


theorem forward_trace (p : ForwardLayout) (length : Nat) (sep : p.x ≠ p.y)
    (u : BlockState) (bounds : RegionBounds)
    (hb : ForwardIO.Bounds p u length bounds) :
    (Kernels.forwardTyped p.s p.x p.l p.y length p.batch p.dim p.B).eraseDType.toAlgKernel.TraceSafe bounds u :=
  forward_trace_of_positive p length (fun _ => sep) u bounds hb


end
end VeriTile.Bench.Repairs.DiagSsmTriton.IOSafety
