import bench.repairs.tritonbench_g.fast_ce_loss.Contract
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.KernelPublic

/-! Complete retained Tensor applications, including actual words/label bytes,
launch/native addresses, scalar ABI observations, every output and full tagged
memory frames. KernelPublic uses KernelTensor and KernelProfile; those proofs
apply the same KernelExecution.*_outputs certificates and KernelMath formulas
exposed by the raw KernelInterface. No source execution is assumed or reproved.
The Real/native domains, scalar ports and compiler choices are unchanged. -/
namespace VeriTile.Bench.Repairs.FastCeLoss.TensorBridge
namespace Real
export KernelPublic.Real (forward chunked backward)
#axiomsClean forward
#axiomsClean chunked
#axiomsClean backward
end Real
namespace Native
export KernelPublic.Native (forward chunked backward)
#axiomsClean forward
#axiomsClean chunked
#axiomsClean backward
end Native
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.FastCeLoss.TensorBridge
