import bench.repairs.tritonbench_g.rope_transform.Misc.NativeCorrect
import bench.repairs.tritonbench_g.rope_transform.Correct
import bench.repairs.tritonbench_g.rope_transform.FPEquiv
import bench.repairs.tritonbench_g.rope_transform.Misc.TensorBridge
import bench.repairs.tritonbench_g.rope_transform.Proofs.KernelAudit

namespace VeriTile.Bench.Repairs.RopeTransform
#axiomsClean Correct.rope_forward_correct
#axiomsClean FPEquiv.rope_forward_fp32
#axiomsClean TensorBridge.rope_forward_kernel_correct
#axiomsClean TensorBridge.rope_forward_kernel_fp32
#axiomsClean TensorBridge.rope_forward_kernel_no_launch
#specNonCircular TensorBridge.rope_forward_kernel_correct avoiding [
  VeriTile.Bench.Repairs.RopeBackwardTransform.Operations.backend]
#specNonCircular TensorBridge.rope_forward_kernel_fp32 avoiding [
  VeriTile.Bench.Repairs.RopeBackwardTransform.Operations.backend]
end VeriTile.Bench.Repairs.RopeTransform
