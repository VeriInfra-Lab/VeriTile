import bench.repairs.tritonbench_g.add_value.Proofs.Contract
import VeriTile.Meta.StatementAudit

/-! Optional Python admission correspondence; primary kernel entries remain independent. -/
namespace VeriTile.Bench.Repairs.AddValue.Contiguous.TensorBridge
noncomputable section

theorem python_checks_iff_preconditions (x : ArgumentMetadata) :
    checkInput x = .ok () ↔ AdmittedArgument x := checkInput_iff x

#axiomsClean python_checks_iff_preconditions
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AddValue.Contiguous.TensorBridge
