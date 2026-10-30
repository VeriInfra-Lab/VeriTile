import bench.repairs.tritonbench_g.sin_computation.Kernels
import bench.tritonbench_g.sin_computation.Kernels

namespace VeriTile.Bench.Repairs.SinComputation.Contiguous
abbrev sourceKernel := VeriTile.Bench.TritonBenchG.SinComputation.sin_kernel
abbrev fp32Kernel (n : Nat) :=
  VeriTile.Bench.TritonBenchG.SinComputation.typedKernel .fp32 .fp32 Bool.false "x_ptr" "output_ptr" n 4
end VeriTile.Bench.Repairs.SinComputation.Contiguous
