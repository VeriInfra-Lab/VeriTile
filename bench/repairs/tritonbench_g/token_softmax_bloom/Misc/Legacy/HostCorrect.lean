import bench.repairs.tritonbench_g.token_softmax_bloom.Kernels
import bench.repairs.tritonbench_g.token_softmax_bloom.Contract
import bench.repairs.tritonbench_g.token_softmax_llama.Misc.HostCorrect

namespace VeriTile.Bench.Repairs.TokenSoftmaxBloom.HostCorrect
export TokenSoftmaxLlama.Correct (realTarget tensor_correct)
#axiomsClean tensor_correct
end VeriTile.Bench.Repairs.TokenSoftmaxBloom.HostCorrect
