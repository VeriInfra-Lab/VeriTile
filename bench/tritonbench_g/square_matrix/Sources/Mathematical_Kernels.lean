import bench.tritonbench_g.square_matrix.Kernels

namespace VeriTile.Bench.TritonBenchG.SquareMatrix
open VeriTile.Triton
abbrev Source := RegionName → RegionName → Nat → Nat → Nat → Nat → ComputeKernel

/-- The actual repaired FP32 source, with every original stride and pointer. -/
def typedKernel (commuted : Bool) : Source :=
  fun out inp si so n B => if commuted then
    VeriTile.Bench.Repairs.SquareMatrix.indexCommutedKernel n B si so out inp
  else VeriTile.Bench.Repairs.SquareMatrix.fp32Kernel n B si so out inp

/-- Mathematical view derived from the shared repair quotation. -/
noncomputable def square_kernel (out inp : RegionName) (si so n B : Nat) : ComputeKernel :=
  let k := (typedKernel Bool.false out inp si so n B).eraseDType.toAlgKernel
  ComputeKernel.fromKernelBody k.inputs k.outputs k.body

end VeriTile.Bench.TritonBenchG.SquareMatrix
