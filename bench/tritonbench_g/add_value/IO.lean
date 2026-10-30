import bench.tritonbench_g.add_value.Proofs.Projection
import VeriTile.Triton.Memory.KernelSpec.Masked

namespace VeriTile.Bench.Repairs.AddValue.Contiguous.IO
open VeriTile.Triton
noncomputable section

/-- The shared literal-add10 source with block size 1024.
This IO describes one program's active input/output lanes in flat memory. -/
def addValueIO (n : Nat) : MaskedKernelIO₁ where
  kernel := originalKernel n
  projection := Projection.original_projection n
  inp := "x"
  out := "out"
  B := blockSize
  read := fun pid => pid * blockSize
  write := fun pid => pid * blockSize
  mask := fun pid i => pid * blockSize + i.val < n

end
end VeriTile.Bench.Repairs.AddValue.Contiguous.IO
