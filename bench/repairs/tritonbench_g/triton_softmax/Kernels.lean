import VeriTile.Triton.DSL

namespace VeriTile.Bench.Repairs.TritonSoftmax
open VeriTile.Triton

/-- Actual repaired source: pointers are allocation regions plus the caller's
storage offsets. Integer int64 casts are erased in this mathematical projection;
the native address theorem retains them. FP32 is the selected default precision.
The original row_max assignment and max/sub/exp/sum/div order are preserved. -/
def softmax_kernel (output_ptr input_ptr : RegionName)
    (input_offset output_offset input_row_stride input_col_stride
      output_row_stride output_col_stride n_cols BLOCK_SIZE : Nat) : ComputeKernel := triton {
  tl.region input_ptr = tl.float32, output_ptr = tl.float32
  row_idx = tl.program_id(axis=0)
  row_start_ptr = input_ptr + $(input_offset) + row_idx * $(input_row_stride)
  out_row_start_ptr = output_ptr + $(output_offset) + row_idx * $(output_row_stride)
  columns = tl.arange(0, $(BLOCK_SIZE))
  row = tl.load(row_start_ptr + columns * $(input_col_stride),
    mask=columns < $(n_cols), other=-float("inf"))
  row_max = tl.max(row, axis=0)
  numerator = tl.exp(row - row_max)
  denominator = tl.sum(numerator, axis=0)
  softmax_output = numerator / denominator
  tl.store(out_row_start_ptr + columns * $(output_col_stride),
    softmax_output, mask=columns < $(n_cols))
}

/-- Genuine integer-address candidate; all FP32 statements and output addresses are unchanged. -/
def indexCommutedKernel (output_ptr input_ptr : RegionName)
    (input_offset output_offset input_row_stride input_col_stride
      output_row_stride output_col_stride n_cols BLOCK_SIZE : Nat) : ComputeKernel := triton {
  tl.region input_ptr = tl.float32, output_ptr = tl.float32
  row_idx = tl.program_id(axis=0)
  row_start_ptr = input_ptr + $(input_offset) + $(input_row_stride) * row_idx
  out_row_start_ptr = output_ptr + $(output_offset) + row_idx * $(output_row_stride)
  columns = tl.arange(0, $(BLOCK_SIZE))
  row = tl.load(row_start_ptr + columns * $(input_col_stride),
    mask=columns < $(n_cols), other=-float("inf"))
  row_max = tl.max(row, axis=0)
  numerator = tl.exp(row - row_max)
  denominator = tl.sum(numerator, axis=0)
  softmax_output = numerator / denominator
  tl.store(out_row_start_ptr + columns * $(output_col_stride),
    softmax_output, mask=columns < $(n_cols))
}

/-- The original wrapper omits num_warps; Triton 3.7.1's default is four at
every width. In particular no softmax1 width-dependent heuristic is inherited. -/
def numWarps (_block : Nat) : Nat := 4
end VeriTile.Bench.Repairs.TritonSoftmax
