import bench.tritonbench_g.rbe_triton_transform.Sources.Kernels

namespace VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch
open VeriTile.Triton
open VeriTile.Bench.Repairs.RbeTritonTransform
noncomputable section

/-- Complete selected FP32 program in the existing mathematical index projection.
The previous adapter erased only nonnegative INT64 widenings, retained signed
Int token arithmetic and made implicit pointer-element FP32 stores explicit.
The full source identity is checked in Misc/SourcePreservation.lean. -/
def typed (dtype : FloatDType)
    (x_ptr out_ptr : RegionName)
    (M K stride_x_batch stride_x_m stride_x_n
      stride_out_batch stride_out_m stride_out_n
      : Nat)
    (start_token_position : Int)
    (THETA : ℝ) (DIM BLOCK_SIZE_M BLOCK_SIZE_K : Nat) :
    ComputeKernel := triton {
  tl.region x_ptr = DTYPE, out_ptr = DTYPE
  M = $((M : Nat))
  K = $((K : Nat))
  stride_x_batch = $((stride_x_batch : Nat))
  stride_x_m = $((stride_x_m : Nat))
  stride_x_n = $((stride_x_n : Nat))
  stride_out_batch = $((stride_out_batch : Nat))
  stride_out_m = $((stride_out_m : Nat))
  stride_out_n = $((stride_out_n : Nat))
  start_token_position = $((start_token_position : Int))
  pid_batch = tl.program_id(axis=0)
  pid = tl.program_id(axis=1)
  pid_m = pid // tl.cdiv(K, $((BLOCK_SIZE_K : Nat)))
  pid_n = pid % tl.cdiv(K, $((BLOCK_SIZE_K : Nat)))

  offs_m = pid_m * $((BLOCK_SIZE_M : Nat)) + tl.arange(0, $((BLOCK_SIZE_M : Nat)))
  offs_n = pid_n * $((BLOCK_SIZE_K : Nat)) + tl.arange(0, $((BLOCK_SIZE_K : Nat)) // $(2)) * $(2)
  x_ptrs = x_ptr + (pid_batch * stride_x_batch + stride_x_m * offs_m[:, None] +
    stride_x_n * offs_n[None, :])
  x_real_mask = (offs_m[:, None] < M) & (offs_n[None, :] < K)
  real = tl.load(x_ptrs, mask=x_real_mask, other=0.0)
  x_imag_mask = (offs_m[:, None] < M) & ($((1 : Nat)) + offs_n[None, :] < K)
  imag = tl.load(x_ptrs + stride_x_n, mask=x_imag_mask, other=0.0)
  tl.debug_barrier()
  start_block = start_token_position + pid_m * $((BLOCK_SIZE_M : Nat))
  freqs = offs_n % $((DIM : Nat))
  freqs_f = tl.toReal(freqs).to(tl.float32) / $((DIM : Nat))
  freqs_p = tl.extra.cuda.libdevice.pow($((THETA : ℝ)), freqs_f)
  tks = tl.arange(0, $((BLOCK_SIZE_M : Nat))) + start_block
  tks_f = tl.toReal(tks)
  freqs_mn = tks_f[:, None] / freqs_p[None, :]
  cos = tl.cos(freqs_mn)
  sin = tl.sin(freqs_mn)

  out_real = real * cos - imag * sin
  out_imag = real * sin + imag * cos
  tl.debug_barrier()
  out_ptrs = out_ptr + (pid_batch * stride_out_batch + stride_out_m * offs_m[:, None] +
    stride_out_n * offs_n[None, :])
  out_real_mask = (offs_m[:, None] < M) & (offs_n[None, :] < K)
  tl.store(out_ptrs, out_real.to(DTYPE), mask=out_real_mask)
  out_imag_mask = (offs_m[:, None] < M) & ($((1 : Nat)) + offs_n[None, :] < K)
  tl.store(out_ptrs + stride_out_n, out_imag.to(DTYPE), mask=out_imag_mask)
}

def source (p : Source.Parameters) : ComputeKernel :=
  typed .fp32 p.input p.output p.rows p.columns p.inputBatchStride p.inputRowStride
    p.inputColumnStride p.outputBatchStride p.outputRowStride p.outputColumnStride
    p.startToken p.theta p.frequencyDimension p.blockRows p.blockColumns

/-- Reuse already computed load masks at the identical output coordinates.
This removes duplicate bounds comparisons without changing floating work. -/
def typedMaskReuse (dtype : FloatDType)
    (x_ptr out_ptr : RegionName)
    (M K stride_x_batch stride_x_m stride_x_n
      stride_out_batch stride_out_m stride_out_n
      : Nat)
    (start_token_position : Int)
    (THETA : ℝ) (DIM BLOCK_SIZE_M BLOCK_SIZE_K : Nat) :
    ComputeKernel := triton {
  tl.region x_ptr = DTYPE, out_ptr = DTYPE
  M = $((M : Nat))
  K = $((K : Nat))
  stride_x_batch = $((stride_x_batch : Nat))
  stride_x_m = $((stride_x_m : Nat))
  stride_x_n = $((stride_x_n : Nat))
  stride_out_batch = $((stride_out_batch : Nat))
  stride_out_m = $((stride_out_m : Nat))
  stride_out_n = $((stride_out_n : Nat))
  start_token_position = $((start_token_position : Int))
  pid_batch = tl.program_id(axis=0)
  pid = tl.program_id(axis=1)
  pid_m = pid // tl.cdiv(K, $((BLOCK_SIZE_K : Nat)))
  pid_n = pid % tl.cdiv(K, $((BLOCK_SIZE_K : Nat)))

  offs_m = pid_m * $((BLOCK_SIZE_M : Nat)) + tl.arange(0, $((BLOCK_SIZE_M : Nat)))
  offs_n = pid_n * $((BLOCK_SIZE_K : Nat)) + tl.arange(0, $((BLOCK_SIZE_K : Nat)) // $(2)) * $(2)
  x_ptrs = x_ptr + (pid_batch * stride_x_batch + stride_x_m * offs_m[:, None] +
    stride_x_n * offs_n[None, :])
  x_real_mask = (offs_m[:, None] < M) & (offs_n[None, :] < K)
  real = tl.load(x_ptrs, mask=x_real_mask, other=0.0)
  x_imag_mask = (offs_m[:, None] < M) & ($((1 : Nat)) + offs_n[None, :] < K)
  imag = tl.load(x_ptrs + stride_x_n, mask=x_imag_mask, other=0.0)
  tl.debug_barrier()
  start_block = start_token_position + pid_m * $((BLOCK_SIZE_M : Nat))
  freqs = offs_n % $((DIM : Nat))
  freqs_f = tl.toReal(freqs).to(tl.float32) / $((DIM : Nat))
  freqs_p = tl.extra.cuda.libdevice.pow($((THETA : ℝ)), freqs_f)
  tks = tl.arange(0, $((BLOCK_SIZE_M : Nat))) + start_block
  tks_f = tl.toReal(tks)
  freqs_mn = tks_f[:, None] / freqs_p[None, :]
  cos = tl.cos(freqs_mn)
  sin = tl.sin(freqs_mn)

  out_real = real * cos - imag * sin
  out_imag = real * sin + imag * cos
  tl.debug_barrier()
  out_ptrs = out_ptr + (pid_batch * stride_out_batch + stride_out_m * offs_m[:, None] +
    stride_out_n * offs_n[None, :])
  out_real_mask = x_real_mask
  tl.store(out_ptrs, out_real.to(DTYPE), mask=out_real_mask)
  out_imag_mask = x_imag_mask
  tl.store(out_ptrs + stride_out_n, out_imag.to(DTYPE), mask=out_imag_mask)
}

def maskReuse (p : Source.Parameters) : ComputeKernel :=
  typedMaskReuse .fp32 p.input p.output p.rows p.columns p.inputBatchStride p.inputRowStride
    p.inputColumnStride p.outputBatchStride p.outputRowStride p.outputColumnStride
    p.startToken p.theta p.frequencyDimension p.blockRows p.blockColumns

end
end VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch
