import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.Adjoint

/-! Algebraic propagation of local numerical residuals. These identities do
not assert any native rounding bound or finite projection for NaN/Inf. Actual
primitive operands and finite-domain obligations are attached in the FP layer.
The saved rounded forward state remains distinct from the ideal trajectory. -/
set_option autoImplicit false
namespace VeriTile.Bench.Repairs.DiagSsmTriton.NumericalPropagation
noncomputable section
open scoped BigOperators

theorem forward_error (state rounded transition input eta : ℂ) :
    (rounded * transition + input + eta) - (state * transition + input) =
      (rounded - state) * transition + eta := by ring

theorem forward_error_step (state rounded next nextRounded transition input eta : ℂ)
    (ideal : next = state * transition + input)
    (actual : nextRounded = rounded * transition + input + eta) :
    nextRounded - next = (rounded - state) * transition + eta := by
  rw [ideal, actual]
  exact forward_error _ _ _ _ _

/-- The real component has a separately rounded second product, followed by
the first-product FMA and final add. No mul/add-to-FMA equality is assumed. -/
theorem complex_real_component_residual (sr si lr li xr eMul eFma eAdd : ℝ) :
    ((sr * lr - (si * li + eMul)) + eFma + xr + eAdd) -
      (sr * lr - si * li + xr) = -eMul + eFma + eAdd := by ring

theorem complex_imag_component_residual (sr si lr li xi eMul eFma eAdd : ℝ) :
    ((sr * li + (si * lr + eMul)) + eFma + xi + eAdd) -
      (sr * li + si * lr + xi) = eMul + eFma + eAdd := by ring

theorem reverse_input_error (gy carry roundedCarry alpha : ℂ) :
    (gy + roundedCarry + alpha) - (gy + carry) = roundedCarry - carry + alpha := by ring

theorem reverse_carry_error (a roundedA transition beta : ℂ) :
    (roundedA * starRingEnd ℂ transition + beta) - (a * starRingEnd ℂ transition) =
      (roundedA - a) * starRingEnd ℂ transition + beta := by ring

/-- The third term explicitly propagates error in saved forward y. Replacing
roundedPrevious with idealPrevious would incorrectly discard this term. -/
theorem reverse_transition_error
    (acc roundedAcc a roundedA idealPrevious roundedPrevious gamma : ℂ) :
    (roundedAcc + roundedA * starRingEnd ℂ roundedPrevious + gamma) -
      (acc + a * starRingEnd ℂ idealPrevious) =
      (roundedAcc - acc) + (roundedA - a) * starRingEnd ℂ idealPrevious +
        roundedA * starRingEnd ℂ (roundedPrevious - idealPrevious) + gamma := by
  simp only [map_sub]
  ring

/-- FP32-to-FP32 byte stores preserve the register word, so there is no extra
store-rounding residual in this selected profile. The equation is stated for
the decoded actual saved word, not for the independent ideal state. -/
theorem saved_state_error (saved rounded ideal : ℂ) (sameWordDecode : saved = rounded) :
    saved - ideal = rounded - ideal := by rw [sameWordDecode]

theorem returned_reduction_error (batch : Nat) (actual ideal : Nat → ℂ) (returned residual : ℂ)
    (localReduction : returned = (Finset.range batch).sum actual + residual) :
    returned - (Finset.range batch).sum ideal =
      (Finset.range batch).sum (fun b => actual b - ideal b) + residual := by
  rw [localReduction, Finset.sum_sub_distrib]
  ring

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DiagSsmTriton.NumericalPropagation
