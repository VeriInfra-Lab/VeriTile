import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceComplexBackwardPrefixSafety
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceComplexBackwardStepSafety
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceComplexBackwardPostSafety
namespace VeriTile.Bench.Repairs.DiagSsmTriton.SourceComplexBackward
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

theorem backward_trace (p : BackwardLayout) (length : Nat) (u : BlockState) (bounds : RegionBounds) :
    (Kernels.complexBackwardTyped p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).eraseDType.toAlgKernel.TraceSafe bounds u ↔
    (Erasure.complexBackward Bool.false p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).toAlgKernel.TraceSafe bounds u := by
  let a := (forwardPrelude p length u).val
  let b := (unsignedPrelude p u).val
  have hi : BackwardRelated p length a b := prelude_related p length u
  obtain ⟨va, vb, ha, hb, hr⟩ := SourceReal.paired_loop "i" length (forwardLoop p length)
    (historicalLoop p length) (BackwardRelated p length)
    (fun i a b ht h => backward_step_related p length i a b ht h)
    0 (Nat.zero_le length) a b hi
  have linked := SourceReal.paired_loop_safe bounds "i" length (forwardLoop p length)
    (historicalLoop p length) (BackwardRelated p length)
    (fun i a b ht h => backward_step_related p length i a b ht h)
    (fun i a b ht h => backward_step_safe p length i a b ht h bounds)
    0 (Nat.zero_le length) a b hi
  simp only [a, Int.natCast_zero] at ha linked
  simp only [b, historicalLoop] at hb linked
  change Stmt.TraceSafeList bounds (forwardBody p length) u ↔ _
  rw [forward_decomposition]
  simp only [Kernel.TraceSafe, ComplexBackwardSource.source_body, List.append_assoc, TraceRewrite.append_iff,
    forward_prefix_safe, (forwardPrelude p length u).property, (unsignedPrelude p u).property,
    Option.some.injEq, forall_eq']
  apply and_congr_right
  intro _
  simp only [Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def, true_and,
    evalOp, forwardPrelude_length, Tile.scalar, Option.bind_some,
    stepStmts, stepForRangeIntAux.forRangeInt_unfold, stepForLoopAux.forLoop_unfold,
    ha, hb, Option.some.injEq, forall_eq', and_true]
  exact and_congr linked (post_safe p length va vb hr bounds)

#axiomsClean backward_trace
end
end VeriTile.Bench.Repairs.DiagSsmTriton.SourceComplexBackward
