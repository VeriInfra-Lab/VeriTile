import bench.repairs.tritonbench_g.sin_computation.Kernels
import VeriTile.Triton.Memory.KernelSpec.Masked

namespace VeriTile.Bench.Repairs.SinComputation.Contiguous.Correct
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₁
noncomputable section

def sinIO (X O : RegionName) (n : Nat) : MaskedKernelIO₁ where
  kernel := kernel X O n blockSize
  projection := by rfl
  inp := X
  out := O
  B := blockSize
  read := fun pid => pid * blockSize
  write := fun pid => pid * blockSize
  mask := fun pid i => pid * blockSize + i.val < n

end
end VeriTile.Bench.Repairs.SinComputation.Contiguous.Correct
