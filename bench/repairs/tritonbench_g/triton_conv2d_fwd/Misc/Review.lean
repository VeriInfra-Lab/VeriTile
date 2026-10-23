import bench.repairs.tritonbench_g.triton_conv2d_fwd.Misc.NativeCorrect
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Correct
import bench.repairs.tritonbench_g.triton_conv2d_fwd.FPEquiv
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Misc.AuxiliaryFPEquiv
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Misc.TensorBridge
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Misc.Reader

namespace VeriTile.Bench.Repairs.Conv2d.Review
#axiomsClean correct_real
#axiomsClean fp_equivalent
#axiomsClean fp_default
#axiomsClean TensorBridge.correct_real
#axiomsClean TensorBridge.fp_equivalent
#axiomsClean TensorBridge.fp_default
#axiomsClean TensorBridge.output_tensor
#axiomsClean KernelNativeResult.physical_execution
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.Conv2d.Review
