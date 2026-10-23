import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.Kernels
import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.Misc.NativeCorrect
import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.Correct
import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.FPEquiv
import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.Misc.TensorBridge

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel
#check @Correct.forward_grid_real
#check @Correct.backward_grid_real
#check @TensorBridge.forward_tensor
#check @TensorBridge.backward_tensor
#axiomsClean Correct.forward_grid_real
#axiomsClean Correct.backward_grid_real
#axiomsClean TensorBridge.forward_tensor
#axiomsClean TensorBridge.backward_tensor
#axiomsClean empty_launch
#axiomsClean output_coverage
#axiomsClean query_coverage
#axiomsClean normalizer_coverage
#specNonCircular backwardMathTarget avoiding [combinedKernel,combinedLaunch,VeriTile.Triton.FP.Structural.exec,runPrograms]
#specNonCircular NativeStage.forwardGridTarget avoiding [VeriTile.Triton.FP.Structural.exec,runPrograms,NativeStage.forwardRun]
#specNonCircular NativeStage.backwardGridTarget avoiding [VeriTile.Triton.FP.Structural.exec,runPrograms,NativeStage.combinedRun]
end VeriTile.Bench.Repairs.ParallelAttentionKernel
