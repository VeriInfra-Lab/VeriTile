import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.Kernels
import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.Contract
import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.Correct
import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.Misc.FPEquivV2
import bench.repairs.tritonbench_g.parallel_attention.Proofs.IONativeAdapter

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel
#check Contract.ShapeContract
#check @Correct.forward_grid_real
#check @Correct.backward_grid_real
#axiomsClean Correct.forward_grid_real
#axiomsClean Correct.backward_grid_real
#specNonCircular NativeStage.forwardGridTarget avoiding [VeriTile.Triton.FP.Structural.exec,runPrograms]
#specNonCircular NativeStage.backwardGridTarget avoiding [VeriTile.Triton.FP.Structural.exec,runPrograms]
#check @FPEquiv.forward_grid
#check @FPEquiv.backward_grid
#axiomsClean FPEquiv.forward_grid
#axiomsClean FPEquiv.backward_grid
#axiomsClean IOData.forward_adapter
#axiomsClean IOData.backward_adapter
#axiomsClean IOData.real_safe
#axiomsClean IOData.native_safe
end VeriTile.Bench.Repairs.ParallelAttentionKernel
