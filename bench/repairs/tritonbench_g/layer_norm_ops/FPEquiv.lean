import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.PairEquivalence

namespace VeriTile.Bench.Repairs.LayerNormOps.FPEquiv
open VeriTile.Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section

abbrev forwardOriginal := @PairIO.forwardOriginal
abbrev forwardMaskReused := @PairIO.forwardMaskReused
abbrev backwardOriginal := @PairIO.backwardOriginal
abbrev backwardRowStartReused := @PairIO.backwardRowStartReused

/-- The complete FP32 forward invocation reuses one column comparison. Every
RMS/LN, residual, residual-output and bias branch keeps its original FP order,
precision, conversions, statistics, outputs and complete exterior frame. -/
specification forward_mask_reused_equiv {α : Type} [Inhabited α]
    (M : Algebra α) (p : ForwardParameters) (pid : Nat)
    (admissible : KernelExecution.ForwardAdmissible p) :
    forwardOriginal M p pid ≡[[]] forwardMaskReused M p pid :=
  PairEquivalence.forward M p pid admissible

/-- The complete bounded backward loop reuses row_start in its upper bound.
The original all-word domain, empty-loop weight read and every optional
recompute/residual/parameter-gradient output remain part of the judgment. -/
specification backward_row_start_reused_equiv {α : Type} [Inhabited α]
    (M : Algebra α) (p : BackwardParameters) (pid : Nat)
    (admissible : KernelReadback.Admissible p) :
    backwardOriginal M p pid ≡[[]] backwardRowStartReused M p pid :=
  PairEquivalence.backward M p pid admissible

#axiomsClean forward_mask_reused_equiv
#axiomsClean backward_row_start_reused_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormOps.FPEquiv
