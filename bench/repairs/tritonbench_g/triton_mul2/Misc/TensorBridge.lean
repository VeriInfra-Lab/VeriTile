import bench.repairs.tritonbench_g.triton_mul2.Correct
import bench.repairs.tritonbench_g.triton_mul2.Proofs.Contract

/-! Optional Python-check correspondence and Tensor argument applications.
The primary mathematical and FP entries do not import this module. -/
namespace VeriTile.Bench.Repairs.TritonMul2.TensorBridge
open VeriTile.Triton VeriTile.Torch
noncomputable section

/-- These are exactly the successful Python admission checks. Storage meaning
is supplied by MemoryContract; no wrapper execution is required. -/
theorem python_checks_iff_preconditions (x : ArgumentMetadata) (B : BlockArgument) :
    checkInputs x B = .ok () ↔ AdmittedArguments x B := checkInputs_iff x B

#axiomsClean python_checks_iff_preconditions
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonMul2.TensorBridge
