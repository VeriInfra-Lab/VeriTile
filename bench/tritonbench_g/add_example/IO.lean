import bench.tritonbench_g.add_example.Proofs.Projection
import VeriTile.Triton.Memory.RegionalKernelIO

/-! Shared canonical add source windows and tail mask. -/
namespace VeriTile.Bench.Repairs.AddExample.Contiguous.IO
open VeriTile.Triton
noncomputable section

def addIO (n : Nat) : MaskedKernelIO₂ where
  kernel := originalKernel n
  projection := Projection.original_projection n
  in1 := "x"
  in2 := "y"
  out := "out"
  B := blockSize
  read1 := fun pid => pid * blockSize
  read2 := fun pid => pid * blockSize
  write := fun pid => pid * blockSize
  mask := fun pid i => pid * blockSize + i.val < n

end
end VeriTile.Bench.Repairs.AddExample.Contiguous.IO
