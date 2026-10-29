import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.RawCorrect
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.RawNative

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.KernelAudit
#axiomsClean RawCorrect.kernel_forward
#axiomsClean RawCorrect.kernel_backward
#axiomsClean RawNative.kernel_forward
#axiomsClean RawNative.kernel_backward
#axiomsClean FPEquiv.zero_bytes
#axiomsClean Kernels.regular_source
#axiomsClean Kernels.gemma_source
#axiomsClean Kernels.input_gradient_source
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
