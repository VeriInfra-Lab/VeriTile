import bench.tritonbench_g.softmax_triton3.Proofs.Mathematical_Kernels
import VeriTile.Triton

namespace VeriTile.Bench.TritonBenchG.SoftmaxTriton3
open VeriTile.Triton

set_option maxRecDepth 16000
set_option maxHeartbeats 3000000

private theorem erased_kernel (k : ComputeKernel)
    (h : k.toAlgorithm? = .ok k.toAlgKernel) :
    k.eraseDType.toAlgKernel = k.toAlgKernel.eraseDType := by
  simp only [ComputeKernel.eraseDType, h, ComputeKernel.toAlgKernel_fromKernelBody]

end VeriTile.Bench.TritonBenchG.SoftmaxTriton3
