import bench.tritonbench_g.matrix_transpose.Kernels

namespace VeriTile.Bench.TritonBenchG.MatrixTranspose
open VeriTile.Triton
abbrev Source := RegionName → RegionName → Nat → Nat → Nat → Nat → Nat → Nat → ComputeKernel

/-- The actual repaired source, including the source's arbitrary strides. -/
def typedKernel (commuted : Bool) : Source :=
  fun M Out sx sy ox oy m d => if commuted then
    VeriTile.Bench.Repairs.MatrixTranspose.indexCommutedSource M Out m d sx sy ox oy
  else VeriTile.Bench.Repairs.MatrixTranspose.source M Out m d sx sy ox oy

/-- Mathematical projection of that same repaired source. -/
noncomputable def kernel (M Out : RegionName) (sx sy ox oy m d : Nat) : ComputeKernel :=
  let k := (typedKernel Bool.false M Out sx sy ox oy m d).eraseDType.toAlgKernel
  ComputeKernel.fromKernelBody k.inputs k.outputs k.body

end VeriTile.Bench.TritonBenchG.MatrixTranspose
