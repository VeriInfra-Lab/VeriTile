import bench.repairs.tritonbench_g.sin_kernel.Proofs.Contract
import bench.repairs.tritonbench_g.sin_kernel.Proofs.Real

namespace VeriTile.Bench.Repairs.SinKernel.Correct
open VeriTile.Triton
noncomputable section

/-- All admitted shapes, including empty/scalar/tail cases. Exact Real.sin is
an independent mathematical interpretation, not native FP32 arithmetic. -/
theorem sin_wrapper_real (x : TensorMetadata) (h : InputContract x)
    (xs : Nat → ℝ) (scratch : BlockState) :
    ∃ final, Math.runWrapper x xs scratch = .ok (some final) ∧
      (∀ i, i < x.numel → final.readMem "output_ptr" i = formula xs i) ∧
      (∀ (r : RegionName) i, r ≠ "output_ptr" ∨ x.numel ≤ i →
        final.mem r i = (Math.allocated xs scratch).mem r i) :=
  Math.wrapper_correct x h xs scratch

#axiomsClean checkInputs_iff
#axiomsClean sin_wrapper_real
#specNonCircular formula avoiding [Math.runWrapper]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SinKernel.Correct
