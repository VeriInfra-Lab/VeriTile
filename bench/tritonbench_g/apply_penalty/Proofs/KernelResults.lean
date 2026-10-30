import bench.tritonbench_g.apply_penalty.Proofs.Rewrite
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.ApplyPenalty.RawFP
open VeriTile.Triton Rewrite
open scoped VeriTile.Spec
noncomputable section

abbrev Rules := FP.ScalarArithmetic.Rules

def original (stride B : Nat) : Program := ⟨stride, B, Bool.false⟩
def reciprocal (stride B : Nat) : Program := ⟨stride, B, Bool.true⟩

/-- The existing reciprocal candidate rewrites only the positive arm. Both
actual typed sources execute successfully; all memory cells and the exterior
frame agree. DIV-MUL-RCP is required only for active lanes selecting that arm.
The wrapper has no finiteness or nonzero-repetition assertion. -/
theorem apply_penalty_reciprocal_equiv (stride B : Nat) (R : Rules) :
    original stride B ≡[R] reciprocal stride B := by
  apply VeriTile.Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  intro α _ M D hM lt hlt s hd
  have hm := reciprocal_memory R M D hM lt hlt stride B s hd
  refine ⟨(originalExecution M lt hlt stride B s).val,
    (reciprocalExecution M lt hlt stride B s).val,
    (originalExecution M lt hlt stride B s).property,
    (reciprocalExecution M lt hlt stride B s).property, hm, ?_⟩
  intro r o outside
  have frame := original_frame M lt hlt stride B s r o outside
  exact ⟨frame, by rw [← hm]; exact frame⟩

/-- The mathematical and floating entries use the same canonical FP32 source. -/
theorem original_source (stride B : Nat) :
    (original stride B).source =
      kernel "logits" "presence" "frequency" "repetition" "ids" "counts" "cumsum" stride 1 B := rfl

theorem reciprocal_source (stride B : Nat) :
    (reciprocal stride B).source =
      reciprocalKernel "logits" "presence" "frequency" "repetition" "ids" "counts" "cumsum" stride 1 B := rfl

#print_fp_assumptions apply_penalty_reciprocal_equiv
#axiomsClean apply_penalty_reciprocal_equiv
#axiomsClean original_source
#axiomsClean reciprocal_source
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ApplyPenalty.RawFP
