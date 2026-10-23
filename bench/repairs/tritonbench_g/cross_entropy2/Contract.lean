import bench.repairs.tritonbench_g.cross_entropy2.Proofs.KernelContract
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.KernelAccess

/-! Kernel-only Tensor arguments and physical storage. The historical checked
wrapper contract is preserved unchanged in Proofs/HostContract.lean. -/
namespace VeriTile.Bench.Repairs.CrossEntropy2.Contract

def sourceSHA256 : String := "628b6d679381ed6d1353fc070a313cc7587ca52676a84eacf16637323c575ba9"

export KernelContract (Word Byte Allocation CoherentBytes TensorStorage FloatBinding LabelBinding
  MatrixView VectorView ScalarOutputView ParametersValid Launch LaunchState
  ForwardViews BackwardViews ForwardInputs BackwardInputs fpValue targetValue wordAt)
export KernelAccess (forward_backed backward_backed native_mask forward_logit_address forward_output_addresses
  int32_widen safe_shift_signed)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.CrossEntropy2.Contract
