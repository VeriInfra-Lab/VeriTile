import bench.repairs.tritonbench_g.sin_computation.Proofs.Contract
import bench.repairs.tritonbench_g.sin_computation.Proofs.Real

namespace VeriTile.Bench.Repairs.SinComputation.Contiguous.Correct
open VeriTile.Triton
noncomputable section

/-- The full guarded host has the independent formula sin(x[i]) and exterior
logical-memory frame. Real semantics do not assert exact native arithmetic. -/
specification sin_wrapper_correct (x out : TensorMetadata) (h : InputContract x out)
    (xs : Nat → ℝ) (scratch : BlockState) :
    Spec.Real (∃ final, Real.runWrapper x out xs scratch = .ok (some final) ∧
      (∀ i, i < x.numel → final.readMem "out" i = formula xs i) ∧
      (∀ (r : RegionName) i, r ≠ "out" ∨ x.numel ≤ i →
        final.mem r i = (Real.allocated xs scratch).mem r i)) :=
  Real.wrapper_correct x out h xs scratch

#axiomsClean checkInputs_iff
#axiomsClean sin_wrapper_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SinComputation.Contiguous.Correct
