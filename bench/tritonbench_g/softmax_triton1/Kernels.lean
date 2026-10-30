import VeriTile.Triton.DSL

namespace VeriTile.Bench.Repairs.SoftmaxTriton1
open VeriTile.Triton

/-- Selected contracted Python source. Allocation regions and storage offsets
represent the actual data pointers. Integer widening is connected to this
index expression in Proofs/Native and Proofs/InputBinding. The source explicitly
retains FP32 loads, subtraction, exp, reductions, division and stores. -/
def softmax_kernel (output_ptr input_ptr : RegionName)
    (input_offset output_offset input_row_stride input_col_stride
      output_row_stride output_col_stride n_cols BLOCK_SIZE : Nat) : ComputeKernel := triton {
  tl.region input_ptr = tl.float32, output_ptr = tl.float32
  row_idx = tl.program_id(0)
  row_start_ptr = input_ptr + $(input_offset) + row_idx * $(input_row_stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  input_ptrs = row_start_ptr + col_offsets * $(input_col_stride)
  row = tl.load(input_ptrs, mask=col_offsets < $(n_cols), other=-float("inf"))
  row_minus_max = row - tl.max(row, axis=0)
  numerator = tl.exp(row_minus_max)
  denominator = tl.sum(numerator, axis=0)
  softmax_output = numerator / denominator
  output_row_start_ptr = output_ptr + $(output_offset) + row_idx * $(output_row_stride)
  output_ptrs = output_row_start_ptr + col_offsets * $(output_col_stride)
  tl.store(output_ptrs, softmax_output, mask=col_offsets < $(n_cols))
}

/-- Existing reciprocal candidate: only the final division is rewritten.
The exp and both reductions retain their original order and precision. -/
def reciprocal_kernel (output_ptr input_ptr : RegionName)
    (input_offset output_offset input_row_stride input_col_stride
      output_row_stride output_col_stride n_cols BLOCK_SIZE : Nat) : ComputeKernel := triton {
  tl.region input_ptr = tl.float32, output_ptr = tl.float32
  row_idx = tl.program_id(0)
  row_start_ptr = input_ptr + $(input_offset) + row_idx * $(input_row_stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  input_ptrs = row_start_ptr + col_offsets * $(input_col_stride)
  row = tl.load(input_ptrs, mask=col_offsets < $(n_cols), other=-float("inf"))
  row_minus_max = row - tl.max(row, axis=0)
  numerator = tl.exp(row_minus_max)
  denominator = tl.sum(numerator, axis=0)
  softmax_output = numerator * (1 / denominator)
  output_row_start_ptr = output_ptr + $(output_offset) + row_idx * $(output_row_stride)
  output_ptrs = output_row_start_ptr + col_offsets * $(output_col_stride)
  tl.store(output_ptrs, softmax_output, mask=col_offsets < $(n_cols))
}


/-- The source wrapper's unchanged launch selection. -/
def numWarps (block : Nat) : Nat := if block ≥ 4096 then 16 else if block ≥ 2048 then 8 else 4

end VeriTile.Bench.Repairs.SoftmaxTriton1
