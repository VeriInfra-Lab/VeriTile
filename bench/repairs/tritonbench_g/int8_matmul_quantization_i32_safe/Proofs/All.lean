import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.ProducerLifecycle
import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Misc.TensorBridge

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe
#axiomsClean Public.TensorBridge.correct
#axiomsClean Public.TensorBridge.fpEquiv
#axiomsClean Public.TensorBridge.composedCorrect
#axiomsClean Public.TensorBridge.composedFPEquiv
#axiomsClean Public.TensorBridge.producerCorrect
#axiomsClean Public.TensorBridge.producerFPEquiv
#axiomsClean Public.correct
#axiomsClean Public.fpEquiv
#axiomsClean Public.composedCorrect
#axiomsClean Public.composedFPEquiv
#axiomsClean Public.producerCorrect
#axiomsClean Public.producerFPEquiv
#axiomsClean gemmValid_iff
#axiomsClean accumulator_native_exact
#axiomsClean Gemm.program_complete
#axiomsClean Grid.source_at_prefix
#axiomsClean Tensor.source_native
#axiomsClean Tensor.native_complete
#axiomsClean Tensor.real_complete
#axiomsClean ProducerTensor.native_complete
#axiomsClean ProducerTensor.real_complete
#axiomsClean ComposedTensor.native_complete
#axiomsClean ComposedTensor.real_complete
#axiomsClean Admission.prepared_binding
#axiomsClean Admission.composed_inputs
#axiomsClean ProducerLifecycle.complete
#axiomsClean Tensor.helperComplete
#axiomsClean Tensor.helperRealComplete
#axiomsClean Tensor.helperNativeComplete
#specNonCircular ComposedTensor.nativeInput avoiding [KernelExecution.exec, ProducerTensor.memory, Tensor.nativeRun]
#specNonCircular ComposedTensor.realInput avoiding [KernelExecution.exec, ProducerTensor.memory, Tensor.realRun]
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe
