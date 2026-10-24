import VeriTile.Triton.DSL
import VeriTile.Triton.DSL.CallSignature

/-! Complete selected original/candidate Python function bodies. Source input
pointers already denote Tensor view bases. The external call supplies their
allocation-region offsets; they are pointer argument values, not extra Python
statements or constexpr parameters. Runtime strides/count preserve their int64
annotations and both original casts. Only BLOCK_SIZE is constexpr. -/
namespace VeriTile.Bench.Repairs.TritonSoftmax
open VeriTile.Triton

def sourceSignature : CallSignature where
  pointers := [("output_ptr",.fp32),("input_ptr",.fp32)]
  scalars := [("input_row_stride",.i64),("input_col_stride",.i64),
    ("output_row_stride",.i64),("output_col_stride",.i64),("n_cols",.i64)]
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["input_ptr"]
  writePorts := ["output_ptr"]

def softmax_kernel (output_ptr input_ptr : RegionName)
    (input_offset output_offset input_row_stride input_col_stride
      output_row_stride output_col_stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  let input_row_stride_value : Int := input_row_stride
  let input_col_stride_value : Int := input_col_stride
  let output_row_stride_value : Int := output_row_stride
  let output_col_stride_value : Int := output_col_stride
  let n_cols_value : Int := n_cols
  with_triton_call sourceSignature call_args
    [input_row_stride := input_row_stride_value, input_col_stride := input_col_stride_value,
     output_row_stride := output_row_stride_value, output_col_stride := output_col_stride_value,
     n_cols := n_cols_value]
    pointer_offsets [input_ptr := input_offset, output_ptr := output_offset] in
  triton {
    row_idx = tl.program_id(axis=0).to(tl.int64)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    out_row_start_ptr = output_ptr + row_idx * output_row_stride
    columns = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    row = tl.load(row_start_ptr + columns * input_col_stride, mask=columns < n_cols, other=-float('inf'))
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    tl.store(out_row_start_ptr + columns * output_col_stride, softmax_output, mask=columns < n_cols)
  }

def indexCommutedKernel (output_ptr input_ptr : RegionName)
    (input_offset output_offset input_row_stride input_col_stride
      output_row_stride output_col_stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  let input_row_stride_value : Int := input_row_stride
  let input_col_stride_value : Int := input_col_stride
  let output_row_stride_value : Int := output_row_stride
  let output_col_stride_value : Int := output_col_stride
  let n_cols_value : Int := n_cols
  with_triton_call sourceSignature call_args
    [input_row_stride := input_row_stride_value, input_col_stride := input_col_stride_value,
     output_row_stride := output_row_stride_value, output_col_stride := output_col_stride_value,
     n_cols := n_cols_value]
    pointer_offsets [input_ptr := input_offset, output_ptr := output_offset] in
  triton {
    row_idx = tl.program_id(axis=0).to(tl.int64)
    row_start_ptr = input_ptr + input_row_stride * row_idx
    out_row_start_ptr = output_ptr + row_idx * output_row_stride
    columns = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    row = tl.load(row_start_ptr + columns * input_col_stride, mask=columns < n_cols, other=-float('inf'))
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    tl.store(out_row_start_ptr + columns * output_col_stride, softmax_output, mask=columns < n_cols)
  }

/-- Retain the selected compiler default of four warps at every width. -/
def numWarps (_block : Nat) : Nat := 4
end VeriTile.Bench.Repairs.TritonSoftmax
