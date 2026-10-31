import VeriTile.Triton.DSL
import VeriTile.Triton.DSL.CallSignature

/-! The readable FP32 source is shared by Real correctness and FP equivalence.
The candidate commutes only the addition; masks, loads and stores are unchanged. -/
namespace VeriTile.Bench.Repairs.VectorAddition.Contiguous
open VeriTile.Triton

def callSignature : CallSignature where
  pointers := [("x_ptr",.fp32),("y_ptr",.fp32),("output_ptr",.fp32)]
  scalars := [("n_elements",.i32)]
  constexpr := ["BLOCK_SIZE"]


def blockSize : Nat := 1024
def gridCount (n : Nat) : Nat := (n + 1023) / 1024

/-- Authoritative original source with explicit FP32 memory and arithmetic tags. -/
def add_kernel (X Y O : RegionName) (n_elements BLOCK_SIZE : Nat) : ComputeKernel :=
  let x_ptr : RegionName := X
  let y_ptr : RegionName := Y
  let output_ptr : RegionName := O
  with_triton_call callSignature call_args [n_elements := n_elements] in
  triton {
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
  }

/-- Existing ADD-COMMUTE candidate, retaining the original statement placement. -/
def add_commuted_kernel (X Y O : RegionName) (n_elements BLOCK_SIZE : Nat) : ComputeKernel :=
  let x_ptr : RegionName := X
  let y_ptr : RegionName := Y
  let output_ptr : RegionName := O
  with_triton_call callSignature call_args [n_elements := n_elements] in
  triton {
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = y + x
    tl.store(output_ptr + offsets, output, mask=mask)
  }

def originalKernel (n : Nat) : ComputeKernel := add_kernel "x" "y" "out" n blockSize

def commutedKernel (n : Nat) : ComputeKernel := add_commuted_kernel "x" "y" "out" n blockSize

end VeriTile.Bench.Repairs.VectorAddition.Contiguous
