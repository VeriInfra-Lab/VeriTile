import bench.repairs.tritonbench_g.masked_add_cuda.Proofs.Contract

/-! Optional Python-check correspondence, separate from kernel specifications. -/
namespace VeriTile.Bench.Repairs.MaskedAddCuda.TensorBridge

theorem python_checks_iff_preconditions (g p m : ArgumentMetadata) (a : AlphaArgument) :
    checkInputs g p m a = .ok () ↔ AdmittedArguments g p m a := checkInputs_iff g p m a

#axiomsClean python_checks_iff_preconditions
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.MaskedAddCuda.TensorBridge
