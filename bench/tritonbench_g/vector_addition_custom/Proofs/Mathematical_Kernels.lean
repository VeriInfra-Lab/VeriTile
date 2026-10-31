import bench.tritonbench_g.vector_addition_custom.Kernels
import VeriTile.Triton

/-! Ordinary mathematical projection of the current repaired source. The DSL
body is maintained only in the canonical Kernels.lean. -/
namespace VeriTile.Bench.TritonBenchG.VectorAdditionCustom
open VeriTile.Triton

def _add_kernel (X Y O : RegionName) (n B : Nat) : ComputeKernel :=
  let k := (VeriTile.Bench.Repairs.VectorAdditionCustom.Contiguous.add_kernel X Y O n B).toAlgKernel
  ComputeKernel.fromKernelBody k.inputs k.outputs k.body

end VeriTile.Bench.TritonBenchG.VectorAdditionCustom
