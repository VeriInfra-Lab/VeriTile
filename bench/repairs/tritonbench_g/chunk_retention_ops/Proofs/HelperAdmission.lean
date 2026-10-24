import bench.repairs.tritonbench_g.chunk_retention_ops.HelperContract
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.Admission
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Contract
theorem kvCheck_iff (a : KV) : kvCheck a=Bool.true ↔ KVContract a := by
  simp [kvCheck,KVContract,shapeCheck_iff,tensorCheck_iff,Bool.and_eq_true,and_assoc]
theorem primalCheck_iff (a : Primal) : primalCheck a=Bool.true ↔ PrimalContract a := by
  simp [primalCheck,PrimalContract,kvCheck_iff,tensorCheck_iff,Bool.and_eq_true]
theorem btCheck_iff (a : BT) : btCheck a=Bool.true ↔ BTContract a := by
  simp [btCheck,BTContract,Bool.and_eq_true]
theorem forwardHCheck_iff (a : ForwardH) : forwardHCheck a=Bool.true ↔ ForwardHContract a := by
  simp [forwardHCheck,ForwardHContract,kvCheck_iff,btCheck_iff,tensorCheck_iff,Bool.and_eq_true,and_assoc]
theorem forwardOCheck_iff (a : ForwardO) : forwardOCheck a=Bool.true ↔ ForwardOContract a := by
  simp [forwardOCheck,ForwardOContract,primalCheck_iff,btCheck_iff,scaleCheck_iff,tensorCheck_iff,
    Bool.and_eq_true,and_assoc]
theorem backwardHCheck_iff (a : BackwardH) : backwardHCheck a=Bool.true ↔ BackwardHContract a := by
  simp [backwardHCheck,BackwardHContract,primalCheck_iff,btCheck_iff,scaleCheck_iff,tensorCheck_iff,
    Bool.and_eq_true,no_overlap_iff,and_assoc]
theorem backwardQKVCheck_iff (a : BackwardQKV) : backwardQKVCheck a=Bool.true ↔ BackwardQKVContract a := by
  simp [backwardQKVCheck,BackwardQKVContract,primalCheck_iff,scaleCheck_iff,tensorCheck_iff,
    Bool.and_eq_true,and_assoc]

#axiomsClean forwardHCheck_iff
#axiomsClean forwardOCheck_iff
#axiomsClean backwardHCheck_iff
#axiomsClean backwardQKVCheck_iff
end VeriTile.Bench.Repairs.ChunkRetentionOps.Contract
