import bench.repairs.tritonbench_g.mixed_sparse_attention.Correct
import bench.repairs.tritonbench_g.mixed_sparse_attention.Misc.NativeCorrect
import bench.repairs.tritonbench_g.mixed_sparse_attention.Misc.TensorBridge

namespace VeriTile.Bench.Repairs.MixedSparseAttention
#auditStmt KernelReal.complete
#auditStmt KernelNative.complete
#axiomsClean kernel_real
#axiomsClean kernel_native
#axiomsClean KernelIO.schedule_run
#axiomsClean KernelIO.whole
#axiomsClean IOSourceSafety.kernel_safe
#axiomsClean IOScope.real_binding
#axiomsClean IOScope.real_domain
#axiomsClean IOScope.native_binding
#axiomsClean IOScope.native_domain
#axiomsClean RawCorrect.kernel_real
#axiomsClean RawNative.kernel_native
#axiomsClean TensorBridge.kernel_real
#axiomsClean TensorBridge.kernel_real_source
#axiomsClean TensorBridge.kernel_native
#axiomsClean TensorBridge.kernel_native_inputs
#axiomsClean TensorBridge.kernel_input_frame
#axiomsClean TensorBridge.kernel_inactive_frame
#axiomsClean TensorBridge.kernel_empty
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.MixedSparseAttention
