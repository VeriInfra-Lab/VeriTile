import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceComplexPrefixSafety
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceComplexStepSafety
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceLoopSimulation
namespace VeriTile.Bench.Repairs.DiagSsmTriton.SourceComplex
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

private theorem empty_tail (o : Option BlockState) :
    (match o with | some _ => True | none => True) := by cases o <;> trivial

/-- Source safety is connected to the existing independent loop without
requiring equal temporary register states. -/
theorem forward_trace (p : ForwardLayout) (length : Nat) (u : BlockState) (bounds : RegionBounds) :
    (Kernels.complexForwardTyped p.s p.x p.y p.l length p.batch p.dim p.B).eraseDType.toAlgKernel.TraceSafe bounds u ↔
    (Erasure.complexForward Bool.false p.s p.x p.y p.l length p.batch p.dim p.B).toAlgKernel.TraceSafe bounds u := by
  change Stmt.TraceSafeList bounds (forwardBody p length) u ↔ _
  rw [forward_decomposition]
  simp only [Kernel.TraceSafe, ComplexForwardSource.source_body, TraceRewrite.append_iff,
    forward_prefix_safe, (forwardPrelude p length u).property, (unsignedPrelude p u).property,
    Option.some.injEq, forall_eq]
  apply and_congr_right
  intro _
  simp only [Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def, true_and,
    evalOp, forwardPrelude_length, Tile.scalar, Option.bind_some, empty_tail, and_true]
  simp only [forall_eq', forwardPrelude_length, Tile.scalar, empty_tail, and_true]
  have linked := VeriTile.Bench.Repairs.DiagSsmTriton.SourceReal.paired_loop_safe bounds "t" length (forwardLoop p length)
    (ComplexForwardSource.body p.x p.y p.batch p.dim p.B)
    (ForwardRelated p length) (fun i a b _ h => forward_step_related p length i a b h)
    (fun i a b _ h => forward_step_safe p length i a b h bounds)
    0 (Nat.zero_le length) _ _ (prelude_related p length u)
  constructor
  · intro h
    exact ⟨linked.mp h.1, by split <;> trivial⟩
  · intro h
    exact ⟨linked.mpr h.1, by split <;> trivial⟩

#axiomsClean forward_trace
end
end VeriTile.Bench.Repairs.DiagSsmTriton.SourceComplex
