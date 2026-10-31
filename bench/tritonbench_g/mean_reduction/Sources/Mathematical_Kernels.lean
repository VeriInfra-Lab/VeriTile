import bench.tritonbench_g.mean_reduction.Kernels

namespace VeriTile.Bench.TritonBenchG.MeanReduction
open VeriTile.Triton
abbrev Source := RegionName → RegionName → Nat → Nat → Nat → Nat → ComputeKernel

/-- The current repaired source and its existing integer-index candidate. -/
def typedKernel (commuted : Bool) : Source :=
  if commuted then VeriTile.Bench.Repairs.MeanReduction.indexCommutedSource
  else VeriTile.Bench.Repairs.MeanReduction.sourceKernel

/-- Mathematical view derived from that same repaired source. -/
noncomputable def mean_dim_kernel (X Mean : RegionName) (M N BM BN : Nat) : ComputeKernel :=
  let k := (typedKernel Bool.false X Mean M N BM BN).eraseDType.toAlgKernel
  ComputeKernel.fromKernelBody k.inputs k.outputs k.body

end VeriTile.Bench.TritonBenchG.MeanReduction
