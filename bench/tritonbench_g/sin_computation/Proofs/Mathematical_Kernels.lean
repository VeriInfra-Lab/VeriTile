import bench.tritonbench_g.sin_computation.Kernels
import VeriTile.Triton

/-! Ordinary mathematical projection of the current repaired source. The DSL
body is maintained only in the canonical Kernels.lean. -/
namespace VeriTile.Bench.TritonBenchG.SinComputation
open VeriTile.Triton

def sin_kernel (X O : RegionName) (n B : Nat) : ComputeKernel :=
  let k := (VeriTile.Bench.Repairs.SinComputation.Contiguous.kernel X O n B).toAlgKernel
  ComputeKernel.fromKernelBody k.inputs k.outputs k.body

end VeriTile.Bench.TritonBenchG.SinComputation
