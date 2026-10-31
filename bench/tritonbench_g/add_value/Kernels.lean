import VeriTile.Triton.DSL
import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.BitValue

/-! Readable explicit-FP32 sources shared by the mathematical, structural
and rewrite proofs. The candidate binds a local constant and swaps operands.
The addition uses the literal 10, matching the Python kernel. The fixed launch
and input contract are unchanged. -/
namespace VeriTile.Bench.Repairs.AddValue.Contiguous
open VeriTile.Triton

def callSignature : CallSignature where
  pointers := [("x_ptr",.fp32),("output_ptr",.fp32)]
  scalars := [("N",.i32)]
  constexpr := ["BLOCK_SIZE"]


def blockSize : Nat := 1024
def gridCount (n : Nat) : Nat := (n + 1023) / 1024
def scalarWord : FP.Value .fp32 := ⟨0x41200000#32⟩
def mathematical (xs : Nat → ℝ) (i : Nat) : ℝ := xs i + 10

/-- Checked `x + 10` source. Typed regions retain explicit FP32 load,
computation and store precision; no handwritten AST is maintained beside it. -/
def kernel (X O : RegionName) (n B : Nat) : ComputeKernel :=
  let x_ptr : RegionName := X
  let output_ptr : RegionName := O
  let BLOCK_SIZE : Nat := B
  with_triton_call callSignature call_args [N := n] in
  triton {
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x + 10
    tl.store(output_ptr + offsets, output, mask=mask)
  }

/-- The actual wrapper selection of the authoritative source. -/
def originalKernel (n : Nat) : ComputeKernel := kernel "x" "out" n blockSize

/-- ADD-COMMUTE candidate. The scalar is a local constant, not a kernel input;
load, mask, addition precision and store match the original source. -/
def commutedKernel (n : Nat) : ComputeKernel :=
  let x_ptr : RegionName := "x"
  let output_ptr : RegionName := "out"
  let BLOCK_SIZE : Nat := blockSize
  with_triton_call callSignature call_args [N := n] in
  triton {
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    y = 10
    output = y + x
    tl.store(output_ptr + offsets, output, mask=mask)
  }

end VeriTile.Bench.Repairs.AddValue.Contiguous
