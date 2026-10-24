import bench.tritonbench_g.fused_recurrent_retention.Kernels
import bench.tritonbench_g.fused_recurrent_retention.Proofs.TypedForward
import bench.tritonbench_g.fused_recurrent_retention.Proofs.TypedBackward
import VeriTile.Triton.Float.ProfiledRewrite

/-! Complete typed state equivalence for both full kernels and every named
common-input/independent-output storage profile. The candidate only reverses
the discrete key-mask comparison. Actual successful source/candidate runs are
proved separately below for fp32; the witness backend is mathematical. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.Equivalence
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def forward (swapped : Bool) (dtype out_dtype : FloatDType) (p : Layout)
    (r : ForwardRegions) (hasSeed storeFinal : Bool) : Program where
  kernel := Full.Program.forward swapped dtype out_dtype p r hasSeed storeFinal
  domain := []

def backward (swapped : Bool) (dtype out_dtype : FloatDType) (p : Layout)
    (r : BackwardRegions) (hasSeed : Bool) : Program where
  kernel := Full.Program.backward swapped dtype out_dtype p r hasSeed
  domain := []

specification fused_recurrent_retention_forward_equiv (dtype out_dtype : FloatDType)
    (p : Layout) (r : ForwardRegions) (hasSeed storeFinal : Bool) :
    forward Bool.false dtype out_dtype p r hasSeed storeFinal ≡[assumptions]
      forward Bool.true dtype out_dtype p r hasSeed storeFinal := by
  apply Spec.FloatingPoint.ofNumerical
    (lhs := forward Bool.false dtype out_dtype p r hasSeed storeFinal)
    (rhs := forward Bool.true dtype out_dtype p r hasSeed storeFinal)
    (structural := fun _ _ => False) rfl (by rfl)
  intro α _ A D _ u _
  exact FPProof.forward_whole_state dtype out_dtype p r hasSeed storeFinal (engine A) u

specification fused_recurrent_retention_backward_equiv (dtype out_dtype : FloatDType)
    (p : Layout) (r : BackwardRegions) (hasSeed : Bool) :
    backward Bool.false dtype out_dtype p r hasSeed ≡[assumptions]
      backward Bool.true dtype out_dtype p r hasSeed := by
  apply Spec.FloatingPoint.ofNumerical
    (lhs := backward Bool.false dtype out_dtype p r hasSeed)
    (rhs := backward Bool.true dtype out_dtype p r hasSeed)
    (structural := fun _ _ => False) rfl (by rfl)
  intro α _ A D _ u _
  exact FPProof.backward_whole_state dtype out_dtype p r hasSeed (engine A) u

theorem fused_recurrent_retention_forward_typed_success {α : Type} [Inhabited α]
    (A : FP.Structural.Algebra α) (p : Layout) (r : ForwardRegions) (hasSeed storeFinal : Bool)
    (u : FP.Structural.State α) :
    ∃ final, FP.Structural.exec A (forward Bool.false .fp32 .fp32 p r hasSeed storeFinal).kernel u = some final ∧
      FP.Structural.exec A (forward Bool.true .fp32 .fp32 p r hasSeed storeFinal).kernel u = some final :=
  TypedExecution.nonempty_pair A p r hasSeed storeFinal u

theorem fused_recurrent_retention_backward_typed_success {α : Type} [Inhabited α]
    (A : FP.Structural.Algebra α) (p : Layout) (r : BackwardRegions) (hasSeed : Bool)
    (u : FP.Structural.State α) :
    ∃ final, FP.Structural.exec A (backward Bool.false .fp32 .fp32 p r hasSeed).kernel u = some final ∧
      FP.Structural.exec A (backward Bool.true .fp32 .fp32 p r hasSeed).kernel u = some final :=
  TypedBackward.nonempty_pair A p r hasSeed u

/-- Explicit nonempty runs start with empty registers, nonzero launch IDs and
both feature tails. Identity casts in this consistency backend are not IEEE. -/
theorem fused_recurrent_retention_forward_nonempty (T : Nat) (hasSeed storeFinal : Bool) :
    ∃ final, FP.Structural.exec (engine TypedExecution.model)
      (forward Bool.false .fp32 .fp32 (TypedExecution.witnessLayout T)
        TypedExecution.witnessRegions hasSeed storeFinal).kernel TypedExecution.witnessInitial = some final ∧
      FP.Structural.exec (engine TypedExecution.model)
      (forward Bool.true .fp32 .fp32 (TypedExecution.witnessLayout T)
        TypedExecution.witnessRegions hasSeed storeFinal).kernel TypedExecution.witnessInitial = some final :=
  TypedExecution.witness_nonempty T hasSeed storeFinal

theorem fused_recurrent_retention_backward_nonempty (T : Nat) (hasSeed : Bool) :
    ∃ final, FP.Structural.exec (engine TypedExecution.model)
      (backward Bool.false .fp32 .fp32 (TypedExecution.witnessLayout T)
        TypedBackward.witnessRegions hasSeed).kernel TypedExecution.witnessInitial = some final ∧
      FP.Structural.exec (engine TypedExecution.model)
      (backward Bool.true .fp32 .fp32 (TypedExecution.witnessLayout T)
        TypedBackward.witnessRegions hasSeed).kernel TypedExecution.witnessInitial = some final :=
  TypedBackward.witness_nonempty T hasSeed

end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.Equivalence
