import bench.tritonbench_g.triton_mul2.Kernels
import VeriTile.Triton

/-! Mathematical views of the current repaired doubling source. These are
ordinary algorithm projections of the shared source, with no second DSL body.
FP equivalence continues to execute the typed source and its current candidate. -/
namespace VeriTile.Bench.TritonBenchG.TritonMul2
open VeriTile.Triton

/-- Ordinary real interpretation of the actual repaired source. -/
def mul2_kernel (X O : RegionName) (n B : Nat) : ComputeKernel :=
  let k := VeriTile.Bench.Repairs.TritonMul2.sourceKernel X O n B |>.toAlgKernel
  ComputeKernel.fromKernelBody k.inputs k.outputs k.body

/-- The same source projection with aliased input and output. -/
def mul2_inplace_kernel (X : RegionName) (n B : Nat) : ComputeKernel :=
  let k := VeriTile.Bench.Repairs.TritonMul2.inplaceSourceKernel X n B |>.toAlgKernel
  ComputeKernel.fromKernelBody k.inputs k.outputs k.body

end VeriTile.Bench.TritonBenchG.TritonMul2
