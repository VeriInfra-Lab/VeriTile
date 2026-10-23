import VeriTile.Triton.DSL

/-! The readable FP32 source is shared by Real correctness and FP equivalence.
The candidate commutes only the addition; masks, loads and stores are unchanged. -/
namespace VeriTile.Bench.Repairs.VectorAdditionCustom.Contiguous
open VeriTile.Triton

def blockSize : Nat := 16
def gridCount (n : Nat) : Nat := (n + 15) / 16

/-- Authoritative original source with explicit FP32 memory and arithmetic tags. -/
def add_kernel (X Y O : RegionName) (n_elements BLOCK_SIZE : Nat) : ComputeKernel :=
  let x_ptr : Region .fp32 := X.cast
  let y_ptr : Region .fp32 := Y.cast
  let output_ptr : Region .fp32 := O.cast
  triton {
    prog_id = tl.program_id(0)
    offs = prog_id * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
    a = tl.load(x_ptr + offs, mask=offs < $(n_elements))
    b = tl.load(y_ptr + offs, mask=offs < $(n_elements))
    tl.store(output_ptr + offs, a + b, mask=offs < $(n_elements))
  }

/-- Existing ADD-COMMUTE candidate, retaining the original statement placement. -/
def add_commuted_kernel (X Y O : RegionName) (n_elements BLOCK_SIZE : Nat) : ComputeKernel :=
  let x_ptr : Region .fp32 := X.cast
  let y_ptr : Region .fp32 := Y.cast
  let output_ptr : Region .fp32 := O.cast
  triton {
    prog_id = tl.program_id(0)
    offs = prog_id * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
    a = tl.load(x_ptr + offs, mask=offs < $(n_elements))
    b = tl.load(y_ptr + offs, mask=offs < $(n_elements))
    tl.store(output_ptr + offs, b + a, mask=offs < $(n_elements))
  }

def originalKernel (n : Nat) : ComputeKernel := add_kernel "x" "y" "out" n blockSize

def commutedKernel (n : Nat) : ComputeKernel := add_commuted_kernel "x" "y" "out" n blockSize

/-- Historical finite-interpreter name; this is the same canonical DSL source. -/
abbrev fp32Kernel (X Y O : RegionName) (n : Nat) : ComputeKernel :=
  add_kernel X Y O n blockSize

end VeriTile.Bench.Repairs.VectorAdditionCustom.Contiguous
