import bench.tritonbench_g.rms_rbe_matmul.Proofs.Contract
import bench.tritonbench_g.rms_matmul_rbe.Proofs.InputContract
namespace VeriTile.Bench.Repairs.RmsRbeMatmul.Contract
open VeriTile.Bench.Repairs.RmsMatmulRbe.Contract

theorem checkInputs_iff (i : Inputs) : checkInputs i=.ok () ↔ InputContract i := by
  classical
  simp [checkInputs,InputContract,ite_eq_iff,tensorGuard_iff,scalarGuard_iff,
    pythonReadFormatGuard_iff,and_assoc]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.RmsRbeMatmul.Contract
