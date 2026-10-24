import bench.repairs.tritonbench_g.context_attn_bloom.Correct
import bench.repairs.tritonbench_g.context_attn_bloom.Misc.NativeCorrect
import bench.repairs.tritonbench_g.context_attn_bloom.FPEquiv
import bench.repairs.tritonbench_g.context_attn_bloom.Misc.TensorBridge

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001
#axiomsClean PublicKernel.correct_real
#axiomsClean PublicKernel.correct_fp32
#axiomsClean KernelIO.implements
#axiomsClean RealKernelIO.implements
#axiomsClean PhysicalConnection.native_schedule
#axiomsClean PhysicalConnection.real_execution
#axiomsClean PhysicalKernel.correct_real
#axiomsClean PhysicalKernel.correct_fp32
#axiomsClean PhysicalKernel.shared_source
#axiomsClean TensorBridge.correct_real
#axiomsClean TensorBridge.correct_fp32
#axiomsClean TensorBridge.input_tensor
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001
