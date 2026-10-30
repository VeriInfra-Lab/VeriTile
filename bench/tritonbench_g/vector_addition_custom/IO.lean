import bench.tritonbench_g.vector_addition_custom.Kernels
import bench.tritonbench_g.vector_addition_custom.Proofs.Projection
import VeriTile.Triton.Memory.RegionalKernelIO

/-! Shared public channel data for the regional primary and optional flat
corollary. The source, precision projection, windows and mask are declared once. -/
namespace VeriTile.Bench.Repairs.VectorAdditionCustom.Contiguous
open VeriTile.Triton
noncomputable section

/-- One program's IO for the canonical source at the wrapper's block size. -/
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
end VeriTile.Bench.Repairs.VectorAdditionCustom.Contiguous
