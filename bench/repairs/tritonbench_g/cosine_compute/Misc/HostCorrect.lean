import bench.repairs.tritonbench_g.cosine_compute.Misc.TensorBridge
import bench.repairs.tritonbench_g.cosine_compute.Proofs.Real

namespace VeriTile.Bench.Repairs.CosineCompute.Correct
open VeriTile.Triton
noncomputable section

/-- Complete original Real source execution at the repaired host grid, with
empty return, every logical output, and the exterior memory frame. -/
specification cosine_wrapper_correct (x : TensorMetadata) (h : InputContract x)
    (input : Nat → ℝ) (scratch : BlockState) :
    Spec.Real (∃ final, Real.runWrapper x input scratch = .ok (some final) ∧
      mathematical x.numel input (final.readMem "b") ∧
      (∀ (r : RegionName) i, r ≠ "b" ∨ x.numel ≤ i → final.mem r i=(Real.allocated input scratch).mem r i)) :=
  Real.wrapper_correct x h input scratch

#axiomsClean TensorBridge.python_checks_iff_preconditions
#axiomsClean cosine_wrapper_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CosineCompute.Correct
