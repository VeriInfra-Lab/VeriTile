import bench.repairs.tritonbench_g.chunk_retention_ops.KernelOnly.Correct
import bench.repairs.tritonbench_g.chunk_retention_ops.KernelOnly.FPEquiv
import bench.repairs.tritonbench_g.chunk_retention_ops.KernelOnly.Misc.TensorBridge

namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelOnly.Review
#axiomsClean Correct.forward_correct
#axiomsClean Correct.output_correct
#axiomsClean Correct.backward_correct
#axiomsClean Correct.gradient_correct
#axiomsClean FPEquiv.kernel_native
#axiomsClean PublicIO.source_safe
#axiomsClean PublicIO.schedule_result
#axiomsClean PublicIO.inputsOfState_binding
#axiomsClean KernelGradient.gradient_read_value_local
#axiomsClean RawCorrect.forward_correct
#axiomsClean RawCorrect.output_correct
#axiomsClean RawCorrect.backward_correct
#axiomsClean RawCorrect.gradient_correct
#axiomsClean RawNative.kernel_native
#axiomsClean TensorBridge.Real.forward_tensor
#axiomsClean TensorBridge.Real.output_tensor
#axiomsClean TensorBridge.Real.backward_tensor
#axiomsClean TensorBridge.Real.gradient_tensor
#axiomsClean TensorBridge.Native.forward_tensor
#axiomsClean TensorBridge.Native.output_tensor
#axiomsClean TensorBridge.Native.backward_tensor
#axiomsClean TensorBridge.Native.gradient_tensor
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelOnly.Review
