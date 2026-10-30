import bench.repairs.tritonbench_g.sin_computation.Correct
import bench.repairs.tritonbench_g.sin_computation.Proofs.Contract

/-! Optional Python-check correspondence and Tensor argument applications.
The primary mathematical and FP entries do not import this module. -/
namespace VeriTile.Bench.Repairs.SinComputation.Contiguous.TensorBridge
open VeriTile.Triton VeriTile.Torch
noncomputable section

theorem python_checks_iff_preconditions (x out : ArgumentMetadata) :
    checkInputs x out = .ok () ↔ AdmittedArguments x out := checkInputs_iff x out

#axiomsClean python_checks_iff_preconditions
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SinComputation.Contiguous.TensorBridge
