import bench.repairs.tritonbench_g.sin_kernel.Proofs.Projection
import VeriTile.Triton.Memory.KernelSpec.Masked

namespace VeriTile.Bench.Repairs.SinKernel.IO
open VeriTile.Triton
noncomputable section

def sinIO (n : Nat) : MaskedKernelIO₁ where
  kernel := originalKernel n
  projection := Projection.original_projection n
  inp := "x_ptr"
  out := "output_ptr"
  B := blockSize
  read := fun pid => pid * blockSize
  write := fun pid => pid * blockSize
  mask := fun pid i => pid * blockSize + i.val < n

end
end VeriTile.Bench.Repairs.SinKernel.IO
