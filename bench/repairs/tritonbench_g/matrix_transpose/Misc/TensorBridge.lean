import bench.repairs.tritonbench_g.matrix_transpose.Correct

/-! Optional Python-check correspondence, separate from the primary kernel
mathematical and equivalence entries. -/
namespace VeriTile.Bench.Repairs.MatrixTranspose.TensorBridge
open VeriTile.Triton
noncomputable section

theorem python_checks_iff_preconditions (m d : Argument) :
    checkInputs m d = .ok () ↔ AdmittedArguments m d := checkInputs_iff m d

#axiomsClean python_checks_iff_preconditions
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatrixTranspose.TensorBridge
