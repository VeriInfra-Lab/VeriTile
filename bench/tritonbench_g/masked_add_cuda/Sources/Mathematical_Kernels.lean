import bench.tritonbench_g.masked_add_cuda.Kernels
/-! Compatibility binding to the sole repaired source family. -/
namespace VeriTile.Bench.TritonBenchG.MaskedAddCuda
open VeriTile.Triton

def integer_kernel (storage : MaskStorage) (candidate : Bool)
    (g p m : RegionName) (n : Nat) (a : ℝ) (B : Nat) : ComputeKernel :=
  if candidate then VeriTile.Bench.Repairs.MaskedAddCuda.commutedKernel storage g p m n a B
  else VeriTile.Bench.Repairs.MaskedAddCuda.kernel storage g p m n a B
end VeriTile.Bench.TritonBenchG.MaskedAddCuda
