import VeriTile.Triton.DSL
import VeriTile.Triton.DSL.CallSignature

/-! The readable FP32 source is shared by Real correctness and FP equivalence.
The candidate commutes only the addition; masks, loads and stores are unchanged. -/
namespace VeriTile.Bench.Repairs.AddExample.Contiguous
open VeriTile.Triton

def callSignature : CallSignature where
  pointers := [("in_ptr0",.fp32),("in_ptr1",.fp32),("out_ptr",.fp32)]
  scalars := [("n_elements",.i32)]
  constexpr := ["BLOCK_SIZE"]


def blockSize : Nat := 4
def gridCount (n : Nat) : Nat := (n + 3) / 4

/-- Authoritative original source with explicit FP32 memory and arithmetic tags. -/
def add_kernel (X Y O : RegionName) (n_elements BLOCK_SIZE : Nat) : ComputeKernel :=
  let in_ptr0 : RegionName := X
  let in_ptr1 : RegionName := Y
  let out_ptr : RegionName := O
  with_triton_call callSignature call_args [n_elements := n_elements] in
  triton {
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    y = tl.load(in_ptr1 + offsets, mask=mask)
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)
  }

/-- Existing ADD-COMMUTE candidate, retaining the original statement placement. -/
def add_commuted_kernel (X Y O : RegionName) (n_elements BLOCK_SIZE : Nat) : ComputeKernel :=
  let in_ptr0 : RegionName := X
  let in_ptr1 : RegionName := Y
  let out_ptr : RegionName := O
  with_triton_call callSignature call_args [n_elements := n_elements] in
  triton {
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    y = tl.load(in_ptr1 + offsets, mask=mask)
    output = y + x
    tl.store(out_ptr + offsets, output, mask=mask)
  }

def originalKernel (n : Nat) : ComputeKernel := add_kernel "x" "y" "out" n blockSize

def commutedKernel (n : Nat) : ComputeKernel := add_commuted_kernel "x" "y" "out" n blockSize

end VeriTile.Bench.Repairs.AddExample.Contiguous
