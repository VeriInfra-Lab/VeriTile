import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Correct
import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Misc.TensorBridge
import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Misc.NativeCorrect

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked
#check @RealCorrectness.attention_kernel_grid
#check @FPEquivalence.attention_native_grid
#check @TensorBridge.attention_fp16
#check @TensorBridge.attention_e5m2
#check @TensorBridge.attention_tensor_backing
#check @FPEquivalence.attention_native_source
#check @TensorBridge.attention_native_tensor
#check @TensorBridge.attention_empty
#axiomsClean RealCorrectness.attention_kernel_grid
#axiomsClean FPEquivalence.attention_native_grid
#axiomsClean TensorBridge.attention_fp16
#axiomsClean TensorBridge.attention_e5m2
#axiomsClean TensorBridge.attention_tensor_backing
#axiomsClean FPEquivalence.attention_native_source
#axiomsClean TensorBridge.attention_native_tensor
#axiomsClean TensorBridge.attention_empty
#axiomsClean Source.execution
#axiomsClean Native.output_binding
#axiomsClean ScalarABI.succeeds_iff
#axiomsClean Backing.caller_observations
#axiomsClean Backing.owner_anchor
#axiomsClean Backing.merge_read
#axiomsClean Backing.typed_binding
#specNonCircular Target.output avoiding [VeriTile.Triton.FP.Structural.exec,VeriTile.Triton.FP.Structural.run,Source.accumulated,Backing.nativePost,Backing.realPost,Backing.merge]
#specNonCircular Tensor.expected avoiding [VeriTile.Triton.FP.Structural.exec,VeriTile.Triton.FP.Structural.run,Source.accumulated,Backing.nativePost,Backing.realPost,Backing.merge]
#specNonCircular RealMathematical.attention avoiding [VeriTile.Triton.exec,VeriTile.Triton.stepStmts,VeriTile.Triton.FP.Structural.exec,VeriTile.Triton.FP.Structural.run,Source.accumulated,Backing.realPost,Backing.merge]
#specNonCircular Mathematical.native avoiding [VeriTile.Triton.FP.Structural.exec,VeriTile.Triton.FP.Structural.run,Source.accumulated,Backing.nativePost,Backing.merge]
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked
