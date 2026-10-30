import VeriTile.Triton.DSL

namespace VeriTile.Bench.Repairs.SoftmaxTriton3
open VeriTile.Triton
/-- The original floating statements, including explicit FP32 casts and mask
addition after input maximum subtraction. All three pointer arguments have
FP32 storage. Addresses are relative to the original Tensor data_ptr; the
checked signed-int64 index sequence is connected in Proofs/Addresses. -/
def softmax_kernel (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) (HAS_MASK : Bool) : ComputeKernel := triton {
  tl.region input_ptr = tl.float32, output_ptr = tl.float32, mask_ptr = tl.float32
  row_idx = tl.program_id(0)
  row_start_ptr = input_ptr + row_idx * $(row_stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  input_ptrs = row_start_ptr + col_offsets
  row = (tl.load(input_ptrs, mask=col_offsets < $(n_cols), other=-float("inf"))).to(tl.float32)
  row_minus_max = row - tl.max(row, axis=0)
  if HAS_MASK {
    mask_ptrs = mask_ptr + row_idx * $(row_stride) + col_offsets
    mask = (tl.load(mask_ptrs, mask=col_offsets < $(n_cols), other=0)).to(tl.float32)
    row_minus_max = row_minus_max + mask
  }
  numerator = tl.exp(row_minus_max)
  denominator = tl.sum(numerator, axis=0)
  softmax_output = numerator / denominator
  output_row_start_ptr = output_ptr + row_idx * $(row_stride)
  output_ptrs = output_row_start_ptr + col_offsets
  tl.store(output_ptrs, softmax_output, mask=col_offsets < $(n_cols))
}

/-- The existing reciprocal transformation with a runtime fallback. Extreme
mask values, nonfinite operands and denominators below one retain original div. -/
def guarded_reciprocal_kernel (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) (HAS_MASK : Bool) : ComputeKernel := triton {
  tl.region input_ptr = tl.float32, output_ptr = tl.float32, mask_ptr = tl.float32
  row_idx = tl.program_id(0)
  row_start_ptr = input_ptr + row_idx * $(row_stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  input_ptrs = row_start_ptr + col_offsets
  row = (tl.load(input_ptrs, mask=col_offsets < $(n_cols), other=-float("inf"))).to(tl.float32)
  row_minus_max = row - tl.max(row, axis=0)
  if HAS_MASK {
    mask_ptrs = mask_ptr + row_idx * $(row_stride) + col_offsets
    mask = (tl.load(mask_ptrs, mask=col_offsets < $(n_cols), other=0)).to(tl.float32)
    row_minus_max = row_minus_max + mask
  }
  numerator = tl.exp(row_minus_max)
  denominator = tl.sum(numerator, axis=0)
  finite_numerator = tl.extra.cuda.libdevice.finitef(numerator)
  finite_count = tl.sum(finite_numerator, axis=0)
  finite_denominator = tl.extra.cuda.libdevice.finitef(denominator)
  if (finite_count == $(BLOCK_SIZE)) & finite_denominator & (denominator >= 1) {
    softmax_output = numerator * (1 / denominator)
  } else {
    softmax_output = numerator / denominator
  }
  output_row_start_ptr = output_ptr + row_idx * $(row_stride)
  output_ptrs = output_row_start_ptr + col_offsets
  tl.store(output_ptrs, softmax_output, mask=col_offsets < $(n_cols))
}

end VeriTile.Bench.Repairs.SoftmaxTriton3
