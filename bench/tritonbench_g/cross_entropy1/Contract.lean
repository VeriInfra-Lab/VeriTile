import bench.tritonbench_g.cross_entropy1.Proofs.KernelContract
import bench.tritonbench_g.cross_entropy1.Proofs.KernelAccess

/-! Kernel-only argument tensors and their physical storage. The original
checked wrapper contract is preserved in Proofs/HostContract.lean. -/
namespace VeriTile.Bench.Repairs.CrossEntropy1.Contract

def sourceSHA256 : String := "e6932518ddf889508a922c49c256fd521eee4041d9115ba2d34954647ba24957"

export KernelContract (Word Byte Allocation CoherentBytes TensorStorage FloatBinding LabelBinding
  MatrixView VectorView ScalarOutputView ParametersValid Launch LaunchState
  ForwardViews BackwardViews ForwardInputs BackwardInputs fpValue targetValue wordAt intAt)
export KernelAccess (forward_backed backward_backed native_mask forward_logit_address forward_output_addresses)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.CrossEntropy1.Contract
