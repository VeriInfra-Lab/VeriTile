import bench.repairs.tritonbench_g.dequantize_rowwise.Proofs.FP32Prior.Kernels
import bench.repairs.tritonbench_g.dequantize_rowwise.Proofs.FP32Prior.Contract
import bench.repairs.tritonbench_g.dequantize_rowwise.Proofs.FP32Prior.Proofs.NativeFull
import bench.repairs.tritonbench_g.dequantize_rowwise.Proofs.FP32Prior.Proofs.Real
namespace VeriTile.Bench.Repairs.DequantizeRowwise.FP32Prior
open VeriTile.Triton FP.Structural
noncomputable section

theorem python_checks_iff_preconditions (x s : ArgumentMetadata) :
    checkInputs x s=.ok () ↔ AdmittedArguments x s := checkInputs_iff x s

/-- Complete ideal-real source/host behavior, including empty inputs, the exact
returned FP16 Tensor object and exterior typed-memory frame. -/
theorem dequantize_rowwise_correct (rest : Algebra ℝ) (c : Direct.Request)
    (out : Direct.Allocation c) (m : Nat → ℝ) :
    ∃ ret,Real.run rest c out m=.ok (some ret) ∧ Real.Result c out m ret :=
  Real.whole rest c out m

#axiomsClean dequantize_rowwise_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DequantizeRowwise.FP32Prior
