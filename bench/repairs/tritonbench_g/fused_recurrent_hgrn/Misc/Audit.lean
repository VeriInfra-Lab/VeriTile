import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Misc.NativeCorrect
import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Correct
import bench.repairs.tritonbench_g.fused_recurrent_hgrn.FPEquiv
import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Misc.TensorBridge
import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.KernelAudit

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn
open VeriTile.Triton
#axiomsClean kernel_correct
#axiomsClean kernel_fp32
#axiomsClean TensorBridge.fused_recurrent_hgrn_correct
#axiomsClean TensorBridge.fused_recurrent_hgrn_fp32
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.FusedRecurrentHgrn
