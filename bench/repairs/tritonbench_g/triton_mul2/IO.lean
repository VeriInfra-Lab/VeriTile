import bench.repairs.tritonbench_g.triton_mul2.Kernels
import VeriTile.Triton.Memory.RegionalKernelIO

/-! Canonical ordinary and in-place windows share one address/mask layout. -/
namespace VeriTile.Bench.Repairs.TritonMul2.IO
open VeriTile.Triton

def mul2IO (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) : MaskedKernelIO₁ where
  kernel := sourceKernel in_ptr0 out_ptr n_elements BLOCK_SIZE
  projection := by rfl
  inp := in_ptr0
  out := out_ptr
  B := BLOCK_SIZE
  read := fun pid => pid * BLOCK_SIZE
  write := fun pid => pid * BLOCK_SIZE
  mask := fun pid j => pid * BLOCK_SIZE + j.val < n_elements

def mul2InplaceIO (ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) : MaskedKernelIO₁ :=
  { mul2IO ptr ptr n_elements BLOCK_SIZE with
    kernel := inplaceSourceKernel ptr n_elements BLOCK_SIZE
    projection := by rfl }

end VeriTile.Bench.Repairs.TritonMul2.IO
