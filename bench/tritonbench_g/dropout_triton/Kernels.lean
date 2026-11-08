import VeriTile.Triton.DSL.CallSignature
import bench.tritonbench_g.dropout_triton.Proofs.Sources.Parameters

/-! Both complete Python bodies are quoted once. Caller metadata supplies
all three floating storage choices and both keep-buffer representations;
the existing original/candidate family keeps all twelve configurations. -/
namespace VeriTile.Bench.Repairs.DropoutTriton
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DropoutTriton.TypedSource

def blockSize : Nat := 1024
def gridCount (n : Nat) : Nat := (n+1023)/1024

def sourceSignature (storage : Storage) (keeping : KeepStorage) : CallSignature where
  pointers := [("x_ptr",match storage with | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16),
    ("x_keep_ptr",match keeping with | .int32 => .i32 | .bool => .boolean),
    ("output_ptr",match storage with | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16)]
  scalars := [("n_elements",.integer),("p",.fp32)]
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["x_ptr","x_keep_ptr"]
  writePorts := ["output_ptr"]

/-- Complete selected original Python body. -/
def quotedOriginalKernel (storage : Storage) (keeping : KeepStorage) (X K O : RegionName) (n B : Nat) (p : ℝ) : ComputeKernel :=
  let x_ptr := X
  let x_keep_ptr := K
  let output_ptr := O
  let BLOCK_SIZE := B
  with_triton_call (sourceSignature storage keeping) call_args [n_elements := n,p := p] in
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
def quotedCommutedKernel (storage : Storage) (keeping : KeepStorage) (X K O : RegionName) (n B : Nat) (p : ℝ) : ComputeKernel :=
  let x_ptr := X
  let x_keep_ptr := K
  let output_ptr := O
  let BLOCK_SIZE := B
  with_triton_call (sourceSignature storage keeping) call_args [n_elements := n,p := p] in
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

def sourceFamily (storage : Storage) (keeping : KeepStorage) (swapped : Bool) :
    RegionName → RegionName → RegionName → Nat → Nat → ℝ → ComputeKernel :=
  if swapped then quotedCommutedKernel storage keeping else quotedOriginalKernel storage keeping

def sourceKernel := sourceFamily .fp32 .int32 Bool.false
def commutedSourceKernel := sourceFamily .fp32 .int32 Bool.true

def kernel (n : Nat) (p : ℝ) : ComputeKernel := sourceKernel "x" "keep" "out" n blockSize p
def indexCommutedKernel (n : Nat) (p : ℝ) : ComputeKernel := commutedSourceKernel "x" "keep" "out" n blockSize p
end VeriTile.Bench.Repairs.DropoutTriton
