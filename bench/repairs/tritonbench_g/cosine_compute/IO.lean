import bench.repairs.tritonbench_g.cosine_compute.Proofs.Projection

/-! Canonical masked cosine kernel input/output window. -/
namespace VeriTile.Bench.Repairs.CosineCompute.IO
open VeriTile.Triton
noncomputable section

def cosine (n : Nat) : MaskedKernelIO₁ where
  kernel := originalKernel n
  projection := Projection.original_projection n
  inp := "a"
  out := "b"
  B := blockSize
  read := fun pid => pid * blockSize
  write := fun pid => pid * blockSize
  mask := fun pid i => pid * blockSize + i.val < n

end
end VeriTile.Bench.Repairs.CosineCompute.IO
