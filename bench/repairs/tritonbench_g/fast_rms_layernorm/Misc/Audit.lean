import bench.repairs.tritonbench_g.fast_rms_layernorm.Correct
import bench.repairs.tritonbench_g.fast_rms_layernorm.Misc.NativeCorrect
import bench.repairs.tritonbench_g.fast_rms_layernorm.Misc.TensorBridge

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.KernelAudit
#axiomsClean TensorBridge.real_kernel_forward
#axiomsClean TensorBridge.real_kernel_backward
#axiomsClean TensorBridge.native_kernel_forward
#axiomsClean TensorBridge.native_kernel_backward
#axiomsClean Correct.kernel_forward
#axiomsClean Correct.kernel_backward
#axiomsClean FPEquiv.kernel_forward
#axiomsClean FPEquiv.kernel_backward
#axiomsClean FPEquiv.zero_bytes
#axiomsClean RealSource.forward_memory
#axiomsClean RealSource.backward_memory
#axiomsClean FPWeightMemory.source_memory
#axiomsClean RealWeightMemory.complete
#axiomsClean NativeWeightSafety.complete
#axiomsClean KernelTensor.logical_read
#axiomsClean Contract.observed_tensor_checks
#axiomsClean Contract.python_checks_iff
#axiomsClean Contract.backward_checks_iff
#axiomsClean NativeGraph.mapWords_shape
#axiomsClean NativeGraph.bits_shape
#axiomsClean NativeGraph.epsilon_shape
#axiomsClean NativeGraph.epsilon_eval
#axiomsClean EpsilonABI.positive_zero
#axiomsClean EpsilonABI.negative_zero
#axiomsClean EpsilonABI.zero_runtime_distinct
#axiomsClean RuntimeForward.Source.runtime_body
#axiomsClean RuntimeForward.Source.runtime_length
#axiomsClean RuntimeForward.Reference.literal_output
#axiomsClean RuntimeForward.Source.forward_memory
#axiomsClean RuntimeForward.Grid.complete
#axiomsClean RuntimeForward.Tensor.forward
#auditStmt EpsilonABI.Binding
#axiomsClean NativeGraph.reduction_complete
#auditStmt NativeGraph.Instructions
#auditStmt NativeGraph.CompilerBinding
#specNonCircular RealReference.output avoiding [Kernels.forward,Kernels.backward,Kernels.weightBackward]
#specNonCircular RealReference.inputGradient avoiding [Kernels.forward,Kernels.backward,Kernels.weightBackward]
#specNonCircular RealReference.weightGradient avoiding [Kernels.forward,Kernels.backward,Kernels.weightBackward]
end VeriTile.Bench.Repairs.FastRmsLayernorm.KernelAudit
