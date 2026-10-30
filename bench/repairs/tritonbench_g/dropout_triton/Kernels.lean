import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.DSL

/-! Complete unchanged Python original and already-existing candidate bodies.
Invocation metadata supplies FP32 data/probability and int32 mask types. No
explicit output cast is added to the pasted source. -/
namespace VeriTile.Bench.Repairs.DropoutTriton
open VeriTile.Triton

def blockSize : Nat := 1024
def gridCount (n : Nat) : Nat := (n + 1023) / 1024

def sourceSignature : CallSignature where
  pointers := [("x_ptr",.fp32),("x_keep_ptr",.i32),("output_ptr",.fp32)]
  scalars := [("n_elements",.integer),("p",.fp32)]
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["x_ptr","x_keep_ptr"]
  writePorts := ["output_ptr"]

/-- Complete selected original Python body. -/
def sourceKernel (X K O : RegionName) (n B : Nat) (p : ℝ) : ComputeKernel :=
  let x_ptr := X
  let x_keep_ptr := K
  let output_ptr := O
  let BLOCK_SIZE := B
  with_triton_call sourceSignature call_args [n_elements := n,p := p] in
  triton {
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    # The line below is the crucial part, described in the paragraph above!
    output = tl.where(x_keep, x / (1 - p), 0.0)
    # Write-back output
    tl.store(output_ptr + offsets, output, mask=mask)
  }

/-- Complete selected pre-existing candidate Python body. -/
def commutedSourceKernel (X K O : RegionName) (n B : Nat) (p : ℝ) : ComputeKernel :=
  let x_ptr := X
  let x_keep_ptr := K
  let output_ptr := O
  let BLOCK_SIZE := B
  with_triton_call sourceSignature call_args [n_elements := n,p := p] in
  triton {
    pid = tl.program_id(axis=0)
    block_start = BLOCK_SIZE * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    # The line below is the crucial part, described in the paragraph above!
    output = tl.where(x_keep, x / (1 - p), 0.0)
    # Write-back output
    tl.store(output_ptr + offsets, output, mask=mask)
  }

def kernel (n : Nat) (p : ℝ) : ComputeKernel :=
  sourceKernel "x" "keep" "out" n blockSize p

def indexCommutedKernel (n : Nat) (p : ℝ) : ComputeKernel :=
  commutedSourceKernel "x" "keep" "out" n blockSize p

end VeriTile.Bench.Repairs.DropoutTriton
