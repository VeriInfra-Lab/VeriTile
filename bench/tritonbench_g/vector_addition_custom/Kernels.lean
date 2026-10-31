import VeriTile.Triton.DSL
import VeriTile.Triton.DSL.CallSignature

/-! The readable FP32 source is shared by Real correctness and FP equivalence.
The candidate commutes only the addition; masks, loads and stores are unchanged. -/
namespace VeriTile.Bench.Repairs.VectorAdditionCustom.Contiguous
open VeriTile.Triton

def callSignature : CallSignature where
  pointers := [("A",.fp32),("B",.fp32),("C",.fp32)]
  scalars := [("size",.i32)]
  constexpr := ["BLOCK"]


def blockSize : Nat := 16
def gridCount (n : Nat) : Nat := (n + 15) / 16

/-- Authoritative original source with explicit FP32 memory and arithmetic tags. -/
def add_kernel (X Y O : RegionName) (n_elements BLOCK_SIZE : Nat) : ComputeKernel :=
  let A : RegionName := X
  let B : RegionName := Y
  let C : RegionName := O
  let BLOCK : Nat := BLOCK_SIZE
  with_triton_call callSignature call_args [size := n_elements] in
  triton {
    """add kernel."""
    prog_id = tl.program_id(0)
    offs = prog_id * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + offs, mask=offs < size)
    b = tl.load(B + offs, mask=offs < size)
    tl.store(C + offs, a + b, mask=offs < size)
  }

/-- Existing ADD-COMMUTE candidate, retaining the original statement placement. -/
def add_commuted_kernel (X Y O : RegionName) (n_elements BLOCK_SIZE : Nat) : ComputeKernel :=
  let A : RegionName := X
  let B : RegionName := Y
  let C : RegionName := O
  let BLOCK : Nat := BLOCK_SIZE
  with_triton_call callSignature call_args [size := n_elements] in
  triton {
    """add kernel."""
    prog_id = tl.program_id(0)
    offs = prog_id * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + offs, mask=offs < size)
    b = tl.load(B + offs, mask=offs < size)
    tl.store(C + offs, b + a, mask=offs < size)
  }

def originalKernel (n : Nat) : ComputeKernel := add_kernel "x" "y" "out" n blockSize

def commutedKernel (n : Nat) : ComputeKernel := add_commuted_kernel "x" "y" "out" n blockSize

/-- Historical finite-interpreter name; this is the same canonical DSL source. -/
abbrev fp32Kernel (X Y O : RegionName) (n : Nat) : ComputeKernel :=
  add_kernel X Y O n blockSize

end VeriTile.Bench.Repairs.VectorAdditionCustom.Contiguous
