import bench.repairs.tritonbench_g.matmul_dequantize_int4.Misc.KernelAudit
import bench.repairs.tritonbench_g.matmul_dequantize_int4.Misc.TensorBridge

namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4
#axiomsClean TensorBridge.Correct.kernel_correct
#axiomsClean TensorBridge.FPEquiv.kernel_equiv
#axiomsClean TensorBridge.Correct.quantize
#axiomsClean TensorBridge.FPEquiv.quantize
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.MatmulDequantizeInt4
