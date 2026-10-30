import bench.tritonbench_g.square_matrix.Kernels
import bench.tritonbench_g.square_matrix.Proofs.Mathematical_Source
import VeriTile.Triton.Memory.KernelSpec.Masked

namespace VeriTile.Bench.Repairs.SquareMatrix.IO
open VeriTile.Triton
noncomputable section

/-- One program owns one contiguous row, with padding excluded by both masks.
This IO uses the same canonical FP32 source as the equivalence proof. -/
def squareIO (n B : Nat) : MaskedKernelIO₁ where
  kernel := (fp32Kernel n B).eraseDType
  projection := VeriTile.Bench.TritonBenchG.SquareMatrix.typed_supported
    .fp32 Bool.false "out" "x" n n n B
  inp := "x"
  out := "out"
  B := B
  read := fun row => row*n
  write := fun row => row*n
  mask := fun _ i => i.val < n

end
end VeriTile.Bench.Repairs.SquareMatrix.IO
