import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Correct
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Misc.RealFacts
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Misc.NativeCorrect
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Misc.TensorBridge
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI
open VeriTile Triton
#axiomsClean Correct.kernel_real
#axiomsClean TensorBridge.tensors_real
#axiomsClean Correct.attention_vjp
#axiomsClean FPEquiv.kernel_native
#axiomsClean TensorBridge.Unbound.tensors_native
#axiomsClean Checked.positive_integer_abi
#axiomsClean Native.Selected.whole_execution_graph
#axiomsClean Native.Selected.whole_execution_words
#axiomsClean TensorStorage.publish_readback
#axiomsClean TensorStorage.publish_frame
#axiomsClean TensorStorage.publish_input
#specNonCircular Native.Target.prepared avoiding [Grid.wholeRun,Native.Selected.run,Native.Selected.stores]
#specNonCircular Native.Target.query avoiding [Source.backwardRounds,Grid.backwardStep,FP.Structural.exec]
#specNonCircular TensorStorage.Objects avoiding [TensorStorage.invokeReal,TensorStorage.invokeNative,FP.Structural.exec]
end VeriTile.Bench.Repairs.TritonAttentionScalarABI
