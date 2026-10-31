import bench.tritonbench_g.sin_kernel.Kernels
import VeriTile.Triton

/-! Ordinary mathematical projection of the current repaired source. The DSL
body is maintained only in the canonical Kernels.lean. -/
namespace VeriTile.Bench.TritonBenchG.SinKernel
open VeriTile.Triton

def kernel_function (X O : RegionName) (n B : Nat) : ComputeKernel :=
  let k := (VeriTile.Bench.Repairs.SinKernel.kernel X O n B).toAlgKernel
  ComputeKernel.fromKernelBody k.inputs k.outputs k.body

end VeriTile.Bench.TritonBenchG.SinKernel
