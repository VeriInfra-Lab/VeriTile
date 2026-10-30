import bench.repairs.tritonbench_g.chunk_retention_ops.KernelOnly.Correct
import bench.repairs.tritonbench_g.chunk_retention_ops.KernelOnly.FPEquiv
import bench.repairs.tritonbench_g.chunk_retention_ops.KernelOnly.Misc.TensorBridge

namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
open VeriTile Triton FP.Structural
#check KernelOnly.Correct.forward_correct
#check KernelOnly.Correct.output_correct
#check KernelOnly.Correct.backward_correct
#check KernelOnly.Correct.gradient_correct
#check KernelOnly.FPEquiv.kernel_native
#axiomsClean KernelOnly.FPEquiv.kernel_native
#check KernelOnly.TensorBridge.Real.forward_tensor
#check KernelOnly.TensorBridge.Real.output_tensor
#check KernelOnly.TensorBridge.Real.backward_tensor
#check KernelOnly.TensorBridge.Real.gradient_tensor
#check KernelOnly.TensorBridge.Native.forward_tensor
#check KernelOnly.TensorBridge.Native.output_tensor
#check KernelOnly.TensorBridge.Native.backward_tensor
#check KernelOnly.TensorBridge.Native.gradient_tensor
#axiomsClean KernelOnly.TensorBridge.Real.forward_tensor
#axiomsClean KernelOnly.TensorBridge.Real.output_tensor
#axiomsClean KernelOnly.TensorBridge.Real.backward_tensor
#axiomsClean KernelOnly.TensorBridge.Real.gradient_tensor
#axiomsClean KernelOnly.TensorBridge.Native.forward_tensor
#axiomsClean KernelOnly.TensorBridge.Native.output_tensor
#axiomsClean KernelOnly.TensorBridge.Native.backward_tensor
#axiomsClean KernelOnly.TensorBridge.Native.gradient_tensor
#check KernelTensorBacking.Buffers
#check KernelTensorBacking.snapshot_native
#check KernelTensorBacking.post_logical
#check KernelTensorBacking.post_input_frame
#check KernelTensorBacking.frame
#axiomsClean KernelTensorBacking.snapshot_native
#axiomsClean KernelTensorBacking.written_unique
#axiomsClean KernelTensorBacking.post_logical
#axiomsClean KernelTensorBacking.post_input_frame
#axiomsClean KernelTensorBacking.frame
#specNonCircular KernelTensorBacking.Buffers avoiding [run,FP.Structural.exec,KernelPacket.reference,KernelNative.target]
#specNonCircular KernelTensorBacking.snapshot avoiding [run,FP.Structural.exec,KernelPacket.reference,KernelNative.target]
#specNonCircular Body.dq avoiding [run,FP.Structural.exec,KernelTensorBacking.post,KernelNative.target]
#specNonCircular Body.dk avoiding [run,FP.Structural.exec,KernelTensorBacking.post,KernelNative.target]
#specNonCircular Body.dvPartial avoiding [run,FP.Structural.exec,KernelTensorBacking.post,KernelNative.target]
#specNonCircular MathTarget.state avoiding [run,FP.Structural.exec,KernelTensorBacking.post]
#specNonCircular MathTarget.future avoiding [run,FP.Structural.exec,KernelTensorBacking.post]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
