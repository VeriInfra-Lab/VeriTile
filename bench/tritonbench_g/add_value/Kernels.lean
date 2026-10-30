import VeriTile.Triton.DSL
import VeriTile.Triton.Float.BitValue

/-! Readable explicit-FP32 sources shared by the mathematical, structural
and rewrite proofs. The candidate binds a local constant and swaps operands.
The addition uses the literal 10, matching the Python kernel. The fixed launch
and input contract are unchanged. -/
namespace VeriTile.Bench.Repairs.AddValue.Contiguous
open VeriTile.Triton

def blockSize : Nat := 1024
def gridCount (n : Nat) : Nat := (n + 1023) / 1024
def scalarWord : FP.Value .fp32 := ⟨0x41200000#32⟩
def mathematical (xs : Nat → ℝ) (i : Nat) : ℝ := xs i + 10

/-- Checked `x + 10` source. Typed regions retain explicit FP32 load,
computation and store precision; no handwritten AST is maintained beside it. -/
def kernel (X O : RegionName) (n B : Nat) : ComputeKernel :=
  let x_ptr : Region .fp32 := X.cast
  let output_ptr : Region .fp32 := O.cast
  triton {
    pid = tl.program_id(axis=0)
    block_start = pid * $(B)
    offsets = block_start + tl.arange(0, $(B))
    mask = offsets < $(n)
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x + 10
    tl.store(output_ptr + offsets, output, mask=mask)
  }

/-- The actual wrapper selection of the authoritative source. -/
def originalKernel (n : Nat) : ComputeKernel := kernel "x" "out" n blockSize

/-- ADD-COMMUTE candidate. The scalar is a local constant, not a kernel input;
load, mask, addition precision and store match the original source. -/
def commutedKernel (n : Nat) : ComputeKernel :=
  let x_ptr : Region .fp32 := ⟨"x"⟩
  let output_ptr : Region .fp32 := ⟨"out"⟩
  triton {
    pid = tl.program_id(axis=0)
    block_start = pid * $(blockSize)
    offsets = block_start + tl.arange(0, $(blockSize))
    mask = offsets < $(n)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = 10
    output = y + x
    tl.store(output_ptr + offsets, output, mask=mask)
  }

end VeriTile.Bench.Repairs.AddValue.Contiguous
