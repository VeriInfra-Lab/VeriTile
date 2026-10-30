import bench.repairs.tritonbench_g.softmax_triton1.Kernels

/-! The original function is exactly the shared softmax_triton1 Python body.
The existing guarded candidate below is pasted without source edits. Both
original casts, all classifiers, the complete runtime fallback and masks remain.
Runtime pointer bases and signed scalar arguments are bound outside the body. -/
namespace VeriTile.Bench.Repairs.SoftmaxTriton2
open VeriTile.Triton

abbrev softmax_kernel := SoftmaxTriton1.softmax_kernel

def sourceSignature : CallSignature where
  pointers := [("output_ptr",.fp32),("input_ptr",.fp32)]
  scalars := [("input_row_stride",.i64),("input_col_stride",.i64),
    ("output_row_stride",.i64),("output_col_stride",.i64),("n_cols",.i64)]
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["input_ptr"]
  writePorts := ["output_ptr"]

def guarded_reciprocal_kernel (output_ptr input_ptr : RegionName)
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
    row_idx = tl.program_id(0).to(tl.int64)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    input_ptrs = row_start_ptr + col_offsets * input_col_stride
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    # Subtract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    finite_numerator = libdevice.finitef(numerator)
    finite_count = tl.sum(finite_numerator, axis=0)
    finite_denominator = libdevice.finitef(denominator)
    if (finite_count == BLOCK_SIZE) & finite_denominator & (denominator >= 1.0):
        softmax_output = numerator * (1.0 / denominator)
    else:
        softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets * output_col_stride
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)
  }

/-- The source wrapper's unchanged warp selection. -/
def numWarps (block : Nat) : Nat := if block ≥ 4096 then 16 else if block ≥ 2048 then 8 else 4

end VeriTile.Bench.Repairs.SoftmaxTriton2
