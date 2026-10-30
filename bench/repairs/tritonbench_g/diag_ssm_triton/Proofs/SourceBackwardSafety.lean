import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceBackwardPrefixSafety
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceBackwardStepSafety
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceBackwardPostSafety
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceBackwardRangeSafety
namespace VeriTile.Bench.Repairs.DiagSsmTriton.SourceBackward
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

theorem backward_trace (p : BackwardLayout) (length : Nat) (u : BlockState) (bounds : RegionBounds) :
    (Kernels.backwardTyped p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).eraseDType.toAlgKernel.TraceSafe bounds u ↔
    (RealReference.diag_ssm_backward_kernel p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).toAlgKernel.TraceSafe bounds u := by
  let a := (forwardPrelude p length u).val
  let b := (unsignedPrelude p u).val
  have hi : BackwardRelated p length a b := prelude_related p length u
  obtain ⟨va, vb, ha, hb, hr⟩ := SourceReal.paired_loop "i" length (forwardLoop p length)
    (historicalLoop p length) (BackwardRelated p length)
    (fun i a b ht h => backward_step_related p length i a b ht h)
    0 (Nat.zero_le length) a b hi
  have hbRange := (BackwardProjection.loop_range "i" 0 length (historicalLoop p length)
    (historicalLoop p length) (fun _ => rfl) b).symm.trans hb
  have linked := SourceReal.paired_loop_safe bounds "i" length (forwardLoop p length)
    (historicalLoop p length) (BackwardRelated p length)
    (fun i a b ht h => backward_step_related p length i a b ht h)
    (fun i a b ht h => backward_step_safe p length i a b ht h bounds)
    0 (Nat.zero_le length) a b hi
  have linkedRange := linked.trans (natural_range_safe bounds "i" 0 length (historicalLoop p length) b)
  simp only [a, Int.natCast_zero] at ha linkedRange
  simp only [b, historicalLoop] at hbRange linkedRange
  change Stmt.TraceSafeList bounds (forwardBody p length) u ↔ _
  rw [forward_decomposition]
  simp only [Kernel.TraceSafe, RealReference.diag_ssm_backward_kernel_toAlg_body,
    RealReference.diagSsmBackwardProjectedBody, List.append_assoc, TraceRewrite.append_iff,
    forward_prefix_safe, (forwardPrelude p length u).property, (unsignedPrelude p u).property,
    Option.some.injEq, forall_eq']
  apply and_congr_right
  intro _
  simp only [Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def, true_and,
    evalOp, forwardPrelude_length, Tile.scalar, Option.bind_some,
    stepStmts, stepForRangeIntAux.forRangeInt_unfold, stepForRangeAux.forRange_unfold,
    ha, hbRange, Option.some.injEq, forall_eq', and_true]
  exact and_congr linkedRange (post_safe p length va vb hr bounds)

#axiomsClean backward_trace
end
end VeriTile.Bench.Repairs.DiagSsmTriton.SourceBackward
