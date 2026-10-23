import bench.repairs.tritonbench_g.sin_kernel.Correct

/-! Optional Tensor applications and Python-check correspondence. -/
namespace VeriTile.Bench.Repairs.SinKernel.TensorBridge
open VeriTile.Triton
noncomputable section

theorem python_checks_iff_preconditions (x : ArgumentMetadata) :
    checkInputs x = .ok () ↔ AdmittedArguments x := checkInputs_iff x
#axiomsClean python_checks_iff_preconditions
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SinKernel.TensorBridge
