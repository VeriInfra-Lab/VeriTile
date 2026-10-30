import bench.repairs.tritonbench_g.relu_triton_kernel.CoordinateModels
import VeriTile.Triton.DSL.CallSignature

/-! Complete, unchanged Python function bodies. Pointer dtypes come from the
call interface; arbitrary-base/stride models used by physical proofs are kept
explicitly in CoordinateModels.lean. -/
namespace VeriTile.Bench.Repairs.ReluTritonKernel
open VeriTile.Triton
def blockSize : Nat := 1024
def gridCount (n : Nat) : Nat := (n + 1023) / 1024
abbrev kernel := Coordinates.kernel
abbrev fusedKernel := Coordinates.fusedKernel

def callSignature : CallSignature where
  pointers := [("x_ptr",.fp32),("out_ptr",.fp32)]
  scalars := []
  constexpr := ["N","block_size"]

def body (x_ptr out_ptr : RegionName) (N block_size : Nat) : ComputeKernel :=
  with_triton_signature callSignature in
  triton {
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    result = tl.where(x >= 0, x, 0.0)
    tl.store(out_ptr + offsets, result, mask=mask)
  }

def inlinedBody (x_ptr out_ptr : RegionName) (N block_size : Nat) : ComputeKernel :=
  with_triton_signature callSignature in
  triton {
    """Existing store-inline candidate; the wrapper selects relu_kernel."""
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, tl.where(x >= 0, x, 0.0), mask=mask)
  }

def originalKernel (N : Nat) : ComputeKernel := body "x" "out" N blockSize

def inlinedKernel (N : Nat) : ComputeKernel := inlinedBody "x" "out" N blockSize

end VeriTile.Bench.Repairs.ReluTritonKernel
