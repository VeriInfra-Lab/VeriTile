import VeriTile.Triton

/-! Existing arbitrary-base/stride coordinate models for the physical proofs.
The selected Python bodies live in Kernels.lean; SourcePreservation proves
their complete execution connection to these canonical element coordinates.
The models retain the existing store-inline candidate.
The original benchmark is kept separately: its pid-zero store gate is defective.
`ib`, `ob`, and `elementBytes` only choose pointer coordinates for optional
physical-memory proofs; the Python kernel uses element offsets (0, 0, 1). -/
namespace VeriTile.Bench.Repairs.ReluTritonKernel.Coordinates
open VeriTile.Triton

def kernel (input output : RegionName) (ib ob elementBytes N B : Nat) : ComputeKernel :=
  let x_ptr : Region .fp32 := input.cast
  let out_ptr : Region .fp32 := output.cast
  triton {
    pid = tl.program_id(0)
    block_start = pid * $(B)
    offsets = block_start + tl.arange(0, $(B))
    mask = offsets < $(N)
    x = tl.load(x_ptr + ($(ib) + $(elementBytes) * offsets), mask=mask)
    result = tl.where(x >= 0, x, 0.0)
    tl.store(out_ptr + ($(ob) + $(elementBytes) * offsets), result, mask=mask)
  }

/-- The same existing store-inline transformation, applied after the gate fix.
Only the private result register disappears; comparison and selection retain
FP32 precision, all input words, and the original masked load/store. -/
def fusedKernel (input output : RegionName) (ib ob elementBytes N B : Nat) : ComputeKernel :=
  let x_ptr : Region .fp32 := input.cast
  let out_ptr : Region .fp32 := output.cast
  triton {
    pid = tl.program_id(0)
    block_start = pid * $(B)
    offsets = block_start + tl.arange(0, $(B))
    mask = offsets < $(N)
    x = tl.load(x_ptr + ($(ib) + $(elementBytes) * offsets), mask=mask)
    tl.store(out_ptr + ($(ob) + $(elementBytes) * offsets), tl.where(x >= 0, x, 0.0), mask=mask)
  }

end VeriTile.Bench.Repairs.ReluTritonKernel.Coordinates
