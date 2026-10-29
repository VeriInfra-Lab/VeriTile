import VeriTile.Triton.DSL.CallSignature

namespace VeriTile.Bench.Repairs.SoftmaxTriton3
open VeriTile.Triton

def callSignature : CallSignature where
  pointers := [("output_ptr",.fp32),("input_ptr",.fp32),("mask_ptr",.fp32)]
  optionalPointers := ["mask_ptr"]
  scalars := [("row_stride",.i64),("n_cols",.i64)]
  constexpr := ["BLOCK_SIZE"]
  readPorts := ["input_ptr","mask_ptr"]
  writePorts := ["output_ptr"]

def softmax_kernel (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) (HAS_MASK : Bool) : ComputeKernel :=
  let row_stride_value : Int := row_stride
  let n_cols_value : Int := n_cols
  with_triton_call callSignature call_args
    [row_stride := row_stride_value, n_cols := n_cols_value]
    pointer_presence [mask_ptr := HAS_MASK] in triton {
    """the kernel function for implementing softmax operator
    Args:
        output_ptr: the output after finishing softmax operation, (N, hidden_dim)
        input_ptr: the tensor of input, shape should be (N, hidden_dim)
        n_cols(tl.constexpr): the number of cols of input
        BLOCK_SIZE(tl.constexpr): the block_size of your hidden_dim dimension, typically BLOCK_SIZE >= hidden_dim
    """
    row_idx = tl.program_id(0).to(tl.int64)
    row_start_ptr = input_ptr + row_idx * row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf')).to(tl.float32)
    row_minus_max = row - tl.max(row, axis=0)
    if mask_ptr is not None:
        mask_ptrs = mask_ptr + row_idx * row_stride + col_offsets
        mask = tl.load(mask_ptrs, mask=col_offsets < n_cols, other=0).to(tl.float32)
        row_minus_max = row_minus_max + mask
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)
  }

def guarded_reciprocal_kernel (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) (HAS_MASK : Bool) : ComputeKernel :=
  let row_stride_value : Int := row_stride
  let n_cols_value : Int := n_cols
  with_triton_call callSignature call_args
    [row_stride := row_stride_value, n_cols := n_cols_value]
    pointer_presence [mask_ptr := HAS_MASK] in triton {
    """the kernel function for implementing softmax operator
    Args:
        output_ptr: the output after finishing softmax operation, (N, hidden_dim)
        input_ptr: the tensor of input, shape should be (N, hidden_dim)
        n_cols(tl.constexpr): the number of cols of input
        BLOCK_SIZE(tl.constexpr): the block_size of your hidden_dim dimension, typically BLOCK_SIZE >= hidden_dim
    """
    row_idx = tl.program_id(0).to(tl.int64)
    row_start_ptr = input_ptr + row_idx * row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf')).to(tl.float32)
    row_minus_max = row - tl.max(row, axis=0)
    if mask_ptr is not None:
        mask_ptrs = mask_ptr + row_idx * row_stride + col_offsets
        mask = tl.load(mask_ptrs, mask=col_offsets < n_cols, other=0).to(tl.float32)
        row_minus_max = row_minus_max + mask
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    finite_numerator = libdevice.finitef(numerator)
    finite_count = tl.sum(finite_numerator, axis=0)
    finite_denominator = libdevice.finitef(denominator)
    if (finite_count == BLOCK_SIZE) & finite_denominator & (denominator >= 1.0):
        softmax_output = numerator * (1.0 / denominator)
    else:
        softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)
  }

end VeriTile.Bench.Repairs.SoftmaxTriton3
