import bench.repairs.tritonbench_g.cosine_compute.Correct
import bench.repairs.tritonbench_g.cosine_compute.Proofs.Contract

/-! Optional Python-check correspondence, separate from the primary kernel
mathematical and equivalence entries. -/
namespace VeriTile.Bench.Repairs.CosineCompute.TensorBridge
open VeriTile.Triton
noncomputable section

theorem python_checks_iff_preconditions (x : ArgumentMetadata) :
    checkInput x = .ok () ↔ AdmittedArgument x := checkInput_iff x

#axiomsClean python_checks_iff_preconditions
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CosineCompute.TensorBridge
