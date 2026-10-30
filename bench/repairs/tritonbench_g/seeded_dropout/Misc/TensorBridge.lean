import bench.repairs.tritonbench_g.seeded_dropout.Proofs.Contract
import bench.repairs.tritonbench_g.seeded_dropout.Correct

/-! Optional Python-check correspondence, separate from the primary kernel
mathematical and equivalence entries. -/
namespace VeriTile.Bench.Repairs.SeededDropout.TensorBridge
open VeriTile.Triton
noncomputable section

theorem python_checks_iff_preconditions (x : VeriTile.Torch.ArgumentMetadata)
    (p : ProbabilityArgument) (seed : SeedArgument) :
    checkInputs x p seed = .ok () ↔ AdmittedArguments x p seed := checkInputs_iff x p seed

#axiomsClean python_checks_iff_preconditions
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SeededDropout.TensorBridge
