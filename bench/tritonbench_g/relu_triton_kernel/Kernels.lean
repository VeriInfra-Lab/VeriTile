import VeriTile.Triton
import VeriTile.Triton.DSL.PointerCells

/-! Canonical readable FP32 repair and the existing store-inline candidate.
The original benchmark is kept separately: its pid-zero store gate is defective.
`ib`, `ob`, and `elementBytes` describe caller-side memory coordinates for
physical-memory proofs; the Python launch uses element coordinates (0, 0, 1).
Both quotations are unchanged Python function bodies. -/
namespace VeriTile.Bench.Repairs.ReluTritonKernel
open VeriTile.Triton

def blockSize : Nat := 1024
def gridCount (n : Nat) : Nat := (n + 1023) / 1024

def sourceSignature : CallSignature where
  pointers := [("x_ptr",.fp32),("out_ptr",.fp32)]
  scalars := []
  constexpr := ["N","block_size"]
  readPorts := ["x_ptr"]
  writePorts := ["out_ptr"]

def sourceKernel (input output : RegionName) (ib ob elementBytes N B : Nat) : ComputeKernel :=
  let x_ptr := input
  let out_ptr := output
  let block_size := B
  with_triton_signature sourceSignature
    pointer_cells [x_ptr := (ib, elementBytes), out_ptr := (ob, elementBytes)] in
  triton {
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    result = tl.where(x >= 0, x, 0.0)
    tl.store(out_ptr + offsets, result, mask=mask)
  }

/-- The same existing store-inline transformation, applied after the gate fix.
Only the private result register disappears; comparison and selection retain
FP32 precision, all input words, and the original masked load/store. -/
def fusedKernel (input output : RegionName) (ib ob elementBytes N B : Nat) : ComputeKernel :=
  let x_ptr := input
  let out_ptr := output
  let block_size := B
  with_triton_signature sourceSignature
    pointer_cells [x_ptr := (ib, elementBytes), out_ptr := (ob, elementBytes)] in
  triton {
    """Existing store-inline candidate; the wrapper selects relu_kernel."""
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, tl.where(x >= 0, x, 0.0), mask=mask)
  }

/-- Source selected by the checked Python wrapper. -/
def originalKernel (N : Nat) : ComputeKernel := sourceKernel "x" "out" 0 0 1 N blockSize

def inlinedKernel (N : Nat) : ComputeKernel := fusedKernel "x" "out" 0 0 1 N blockSize

end VeriTile.Bench.Repairs.ReluTritonKernel
