import bench.repairs.tritonbench_g.kldiv_compute.Proofs.ConditionalCorrect
import bench.repairs.tritonbench_g.kldiv_compute.Proofs.Checked.Real

namespace VeriTile.Bench.Repairs.KldivCompute.Contiguous.Correct
open VeriTile.Triton FP.Structural
noncomputable section

theorem python_checks_iff_preconditions (x y : ArgumentMetadata) :
    checkInputs x y=.ok () ↔ AdmittedArguments x y := checkInputs_iff x y

/-- Whole checked logical output and exterior memory frame in the independent
Real interpretation, including arbitrary rank, scalar shapes and empties. -/
theorem checked_real (x y : TensorMetadata) (h : InputContract x y)
    (xs ys : Nat → ℝ) (scratch : State ℝ) :
    ∃ final,RealProof.runWrapper x y xs ys scratch=.ok (some final) ∧
      (∀ i,i<x.numel → final.mem "out" i=.mk .real (mathematical (xs i) (ys i))) ∧
      (∀ r i,r≠"out" ∨ x.numel≤ i → final.mem r i=(RealProof.allocated xs ys scratch).mem r i) :=
  RealProof.wrapper_correct x y h xs ys scratch

#axiomsClean python_checks_iff_preconditions
#axiomsClean checked_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivCompute.Contiguous.Correct
