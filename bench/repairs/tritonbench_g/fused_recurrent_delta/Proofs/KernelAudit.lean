import bench.repairs.tritonbench_g.fused_recurrent_delta.Whole
import bench.repairs.tritonbench_g.fused_recurrent_delta.Misc.TensorBridge
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta
#axiomsClean kernel_checked
#axiomsClean kernel_scale_projection
#axiomsClean TensorBridge.kernel_forward_real
#axiomsClean TensorBridge.kernel_backward_real
#axiomsClean kernel_forward_programs
#axiomsClean kernel_saved_from_forward
#axiomsClean kernel_key_padding
#axiomsClean kernel_full_tensor_vjp
#axiomsClean kernel_scalar_beta_vjp
#axiomsClean TensorBridge.kernel_native_tensor
#axiomsClean kernel_native_source
#axiomsClean kernel_runtime_scale
#axiomsClean kernel_positive_underflow
#axiomsClean TensorBridge.kernel_physical_logical_input
#axiomsClean KernelTensor.native_input
#axiomsClean kernel_pointer_projection
#axiomsClean kernel_formed_indices
#axiomsClean kernel_reverse_retreat
#axiomsClean kernel_program_separation
#axiomsClean kernel_forward_coverage
#axiomsClean kernel_forward_state_coverage
#axiomsClean kernel_backward_key_coverage
#axiomsClean kernel_backward_value_coverage
#axiomsClean kernel_backward_state_coverage
#axiomsClean kernel_scalar_beta_coverage
#axiomsClean kernel_vector_beta_coverage
#specNonCircular KernelCheck.checkInput avoiding [VeriTile.Triton.FP.Structural.exec,VeriTile.Triton.FP.Structural.run,KernelProgram.source]
#specNonCircular KernelTensor.Buffers avoiding [VeriTile.Triton.FP.Structural.exec,VeriTile.Triton.FP.Structural.run,KernelProgram.source]
#specNonCircular KernelCompiler.Binding avoiding [VeriTile.Triton.FP.Structural.exec,VeriTile.Triton.FP.Structural.run]
#specNonCircular KernelCompiler.Expected avoiding [VeriTile.Triton.FP.Structural.exec,VeriTile.Triton.FP.Structural.run,KernelProgram.source]
#specNonCircular KernelReal.ForwardExpected avoiding [VeriTile.Triton.FP.Structural.exec,VeriTile.Triton.FP.Structural.run,KernelProgram.source,KernelReference.forward]
#specNonCircular KernelReal.BackwardExpected avoiding [VeriTile.Triton.FP.Structural.exec,VeriTile.Triton.FP.Structural.run,KernelProgram.source,KernelBackwardReference.reverse]
#specNonCircular KernelNativeSpec.forward avoiding [VeriTile.Triton.FP.Structural.exec,VeriTile.Triton.FP.Structural.run,KernelProgram.source,KernelReference.forward]
#specNonCircular KernelNativeBackwardSpec.reverse avoiding [VeriTile.Triton.FP.Structural.exec,VeriTile.Triton.FP.Structural.run,KernelProgram.source,KernelBackwardReference.reverse]
#specNonCircular KernelNativeBackwardSpec.correction avoiding [VeriTile.Triton.FP.Structural.exec,VeriTile.Triton.FP.Structural.run,KernelProgram.source,KernelBackwardReference.correction]
#specNonCircular KernelReal.Saved avoiding [VeriTile.Triton.FP.Structural.exec,VeriTile.Triton.FP.Structural.run,KernelProgram.source]
#specNonCircular KernelRealBound.ColumnVJP avoiding [VeriTile.Triton.FP.Structural.exec,VeriTile.Triton.FP.Structural.run,KernelProgram.source]
end VeriTile.Bench.Repairs.FusedRecurrentDelta
