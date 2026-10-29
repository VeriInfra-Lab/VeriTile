import bench.repairs.tritonbench_g.sin_kernel.Kernels
import bench.tritonbench_g.sin_kernel.Kernels

namespace VeriTile.Bench.Repairs.SinKernel
abbrev sourceKernel := VeriTile.Bench.TritonBenchG.SinKernel.kernel_function
abbrev fp32Kernel (n : Nat) :=
  VeriTile.Bench.TritonBenchG.SinKernel.typedKernel .fp32 Bool.false "x_ptr" "output_ptr" n 1024
end VeriTile.Bench.Repairs.SinKernel
