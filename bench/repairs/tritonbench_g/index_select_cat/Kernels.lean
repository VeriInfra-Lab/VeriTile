import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.DSL
import VeriTile.Triton.Source

/-! Complete selected original and existing Python candidate; signed widening
casts, source metadata, pointer order and all masks remain in the raw bodies. -/
namespace VeriTile.Bench.Repairs.IndexSelectCat
open VeriTile.Triton

structure Layout where
  (output source index : RegionName)
  (rows cols rowStride colStride BI BC : Nat)

def sourceSignature : CallSignature where
  pointers := [("output_ptr",.fp32), ("source_ptr",.fp32), ("index_ptr",.i64)]
  scalars := [("num_indices",.integer), ("num_cols",.integer), ("stride0",.integer), ("stride1",.integer)]
  constexpr := ["BLOCK_SIZE_INDEX","BLOCK_SIZE_COL"]
  readPorts := ["source_ptr","index_ptr"]
  writePorts := ["output_ptr"]

/-- Complete actual original Python body. -/
def kernel (p : Layout) : ComputeKernel :=
  let output_ptr := p.output
  let source_ptr := p.source
  let index_ptr := p.index
  let BLOCK_SIZE_INDEX := p.BI
  let BLOCK_SIZE_COL := p.BC
  with_triton_call sourceSignature call_args [num_indices := p.rows, num_cols := p.cols, stride0 := p.rowStride, stride1 := p.colStride] in
  triton {
    pid0 = tl.program_id(axis=0).to(tl.int64)
    pid1 = tl.program_id(axis=1).to(tl.int64)
    indices = pid0 * BLOCK_SIZE_INDEX + tl.arange(0, BLOCK_SIZE_INDEX).to(tl.int64)
    rows = tl.load(index_ptr + indices, mask=indices < num_indices, other=0).to(tl.int64)
    cols = pid1 * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL).to(tl.int64)
    source_offsets = source_ptr + rows[:, None] * stride0 + cols[None, :] * stride1
    mask = (indices[:, None] < num_indices) & (cols[None, :] < num_cols)
    output = tl.load(source_offsets, mask=mask)
    output_offsets = output_ptr + indices[:, None] * stride0 + cols[None, :] * stride1
    tl.store(output_offsets, output, mask=mask)
  }

/-- Complete actual pre-existing candidate Python body. -/
def indexCommutedKernel (p : Layout) : ComputeKernel :=
  let output_ptr := p.output
  let source_ptr := p.source
  let index_ptr := p.index
  let BLOCK_SIZE_INDEX := p.BI
  let BLOCK_SIZE_COL := p.BC
  with_triton_call sourceSignature call_args [num_indices := p.rows, num_cols := p.cols, stride0 := p.rowStride, stride1 := p.colStride] in
  triton {
    pid0 = tl.program_id(axis=0).to(tl.int64)
    pid1 = tl.program_id(axis=1).to(tl.int64)
    indices = BLOCK_SIZE_INDEX * pid0 + tl.arange(0, BLOCK_SIZE_INDEX).to(tl.int64)
    rows = tl.load(index_ptr + indices, mask=indices < num_indices, other=0).to(tl.int64)
    cols = pid1 * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL).to(tl.int64)
    source_offsets = source_ptr + rows[:, None] * stride0 + cols[None, :] * stride1
    mask = (indices[:, None] < num_indices) & (cols[None, :] < num_cols)
    output = tl.load(source_offsets, mask=mask)
    output_offsets = output_ptr + indices[:, None] * stride0 + cols[None, :] * stride1
    tl.store(output_offsets, output, mask=mask)
  }

end VeriTile.Bench.Repairs.IndexSelectCat
