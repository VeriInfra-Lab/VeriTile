import bench.repairs.tritonbench_g.token_softmax_bloom.Kernels
import bench.repairs.tritonbench_g.token_softmax_bloom.Contract
import bench.repairs.tritonbench_g.token_softmax_llama.Misc.HostFPEquiv

namespace VeriTile.Bench.Repairs.TokenSoftmaxBloom.HostFPEquiv
export TokenSoftmaxLlama.FPEquiv (nativeTarget tensor_fp32 local_refinement)
#axiomsClean tensor_fp32
#axiomsClean local_refinement
end VeriTile.Bench.Repairs.TokenSoftmaxBloom.HostFPEquiv
