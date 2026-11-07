import bench.tritonbench_g.rbe_triton_transform.Proofs.Sources.Kernels
import VeriTile.Triton.DSL.StatementJitFunctions

namespace VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch
open VeriTile.Triton
open VeriTile.Bench.Repairs.RbeTritonTransform
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

/-- Complete source helper. Its local DIM is fixed by the Python body. -/
def frequencySource : StatementJitFunction :=
  with_triton_params (offs_cn, starting_idx, theta, NB_TOKENS) constexpr_args [theta, NB_TOKENS] in triton {
  DIM: tl.constexpr = 128  # in model, dim = self.params.dim // self.params.n_heads
  freqs = offs_cn % DIM
  freqs = freqs.to(tl.float32) / DIM
  freqs = tl.extra.cuda.libdevice.pow(theta, freqs)
  freqs = (tl.arange(0, NB_TOKENS).to(tl.int64) + tl.cast(starting_idx, tl.int64))[:, None] / freqs[None, :]
  return tl.cos(freqs), tl.sin(freqs)
  }

def callSignature (dtype : FloatDType) : CallSignature where
  floatingPointers := [("x_ptr",dtype),("out_ptr",dtype)]
  scalars := [("M",.integer),("K",.integer),("stride_x_batch",.integer),
    ("stride_x_m",.integer),("stride_x_n",.integer),("stride_out_batch",.integer),
    ("stride_out_m",.integer),("stride_out_n",.integer),("start_token_position",.integer)]
  constexpr := ["THETA","BLOCK_SIZE_M","BLOCK_SIZE_K"]
  readPorts := ["x_ptr"]
  writePorts := ["out_ptr"]

def typedKernel (dtype : FloatDType) (x_ptr out_ptr : RegionName)
    (M K stride_x_batch stride_x_m stride_x_n stride_out_batch stride_out_m
      stride_out_n : Nat) (start_token_position : Int) (THETA : ℝ)
    (BLOCK_SIZE_M BLOCK_SIZE_K : Nat) : ComputeKernel :=
  with_triton_call (callSignature dtype) call_args [M := M, K := K,
    stride_x_batch := stride_x_batch, stride_x_m := stride_x_m, stride_x_n := stride_x_n,
    stride_out_batch := stride_out_batch, stride_out_m := stride_out_m,
    stride_out_n := stride_out_n, start_token_position := start_token_position]
    jit_functions [get_freq_multi_tokens := frequencySource] in triton {
  M = tl.cast(M, tl.int64)
  K = tl.cast(K, tl.int64)
  stride_x_batch = tl.cast(stride_x_batch, tl.int64)
  stride_x_m = tl.cast(stride_x_m, tl.int64)
  stride_x_n = tl.cast(stride_x_n, tl.int64)
  stride_out_batch = tl.cast(stride_out_batch, tl.int64)
  stride_out_m = tl.cast(stride_out_m, tl.int64)
  stride_out_n = tl.cast(stride_out_n, tl.int64)
  start_token_position = tl.cast(start_token_position, tl.int64)
  pid_batch = tl.program_id(axis=0).to(tl.int64)
  pid = tl.program_id(axis=1).to(tl.int64)
  pid_m = pid // tl.cdiv(K, BLOCK_SIZE_K)
  pid_n = pid % tl.cdiv(K, BLOCK_SIZE_K)

  offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
  offs_n = pid_n * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K // 2).to(tl.int64) * 2
  x_ptrs = x_ptr + (pid_batch * stride_x_batch + stride_x_m * offs_m[:, None] + stride_x_n * offs_n[None, :])
  x_real_mask = (offs_m[:, None] < M) & (offs_n[None, :] < K)
  real = tl.load(x_ptrs, mask=x_real_mask, other=0.0)
  x_imag_mask = (offs_m[:, None] < M) & (1 + offs_n[None, :] < K)
  imag = tl.load(x_ptrs + stride_x_n, mask=x_imag_mask, other=0.0)
  tl.debug_barrier()
  start_block = start_token_position + pid_m * BLOCK_SIZE_M
  cos, sin = get_freq_multi_tokens(offs_cn=offs_n, starting_idx=start_block, theta=THETA, NB_TOKENS=BLOCK_SIZE_M)

  out_real = real * cos - imag * sin
  out_imag = real * sin + imag * cos
  tl.debug_barrier()
  out_ptrs = out_ptr + (
          pid_batch * stride_out_batch + stride_out_m * offs_m[:, None] + stride_out_n * offs_n[None, :])
  out_real_mask = (offs_m[:, None] < M) & (offs_n[None, :] < K)
  tl.store(out_ptrs, out_real, mask=out_real_mask)
  out_imag_mask = (offs_m[:, None] < M) & (1 + offs_n[None, :] < K)
  tl.store(out_ptrs + stride_out_n, out_imag, mask=out_imag_mask)
  }

def source (p : Source.Parameters) : ComputeKernel :=
  typedKernel .fp32 p.input p.output p.rows p.columns p.inputBatchStride p.inputRowStride
    p.inputColumnStride p.outputBatchStride p.outputRowStride p.outputColumnStride
    p.startToken p.theta p.blockRows p.blockColumns

def typedMaskReuseKernel (dtype : FloatDType) (x_ptr out_ptr : RegionName)
    (M K stride_x_batch stride_x_m stride_x_n stride_out_batch stride_out_m
      stride_out_n : Nat) (start_token_position : Int) (THETA : ℝ)
    (BLOCK_SIZE_M BLOCK_SIZE_K : Nat) : ComputeKernel :=
  with_triton_call (callSignature dtype) call_args [M := M, K := K,
    stride_x_batch := stride_x_batch, stride_x_m := stride_x_m, stride_x_n := stride_x_n,
    stride_out_batch := stride_out_batch, stride_out_m := stride_out_m,
    stride_out_n := stride_out_n, start_token_position := start_token_position]
    jit_functions [get_freq_multi_tokens := frequencySource] in triton {
  M = tl.cast(M, tl.int64)
  K = tl.cast(K, tl.int64)
  stride_x_batch = tl.cast(stride_x_batch, tl.int64)
  stride_x_m = tl.cast(stride_x_m, tl.int64)
  stride_x_n = tl.cast(stride_x_n, tl.int64)
  stride_out_batch = tl.cast(stride_out_batch, tl.int64)
  stride_out_m = tl.cast(stride_out_m, tl.int64)
  stride_out_n = tl.cast(stride_out_n, tl.int64)
  start_token_position = tl.cast(start_token_position, tl.int64)
  pid_batch = tl.program_id(axis=0).to(tl.int64)
  pid = tl.program_id(axis=1).to(tl.int64)
  pid_m = pid // tl.cdiv(K, BLOCK_SIZE_K)
  pid_n = pid % tl.cdiv(K, BLOCK_SIZE_K)

  offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
  offs_n = pid_n * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K // 2).to(tl.int64) * 2
  x_ptrs = x_ptr + (pid_batch * stride_x_batch + stride_x_m * offs_m[:, None] + stride_x_n * offs_n[None, :])
  x_real_mask = (offs_m[:, None] < M) & (offs_n[None, :] < K)
  real = tl.load(x_ptrs, mask=x_real_mask, other=0.0)
  x_imag_mask = (offs_m[:, None] < M) & (1 + offs_n[None, :] < K)
  imag = tl.load(x_ptrs + stride_x_n, mask=x_imag_mask, other=0.0)
  tl.debug_barrier()
  start_block = start_token_position + pid_m * BLOCK_SIZE_M
  cos, sin = get_freq_multi_tokens(offs_cn=offs_n, starting_idx=start_block, theta=THETA, NB_TOKENS=BLOCK_SIZE_M)

  out_real = real * cos - imag * sin
  out_imag = real * sin + imag * cos
  tl.debug_barrier()
  out_ptrs = out_ptr + (
          pid_batch * stride_out_batch + stride_out_m * offs_m[:, None] + stride_out_n * offs_n[None, :])
  out_real_mask = x_real_mask
  tl.store(out_ptrs, out_real, mask=out_real_mask)
  out_imag_mask = x_imag_mask
  tl.store(out_ptrs + stride_out_n, out_imag, mask=out_imag_mask)
  }

def maskReuse (p : Source.Parameters) : ComputeKernel :=
  typedMaskReuseKernel .fp32 p.input p.output p.rows p.columns p.inputBatchStride p.inputRowStride
    p.inputColumnStride p.outputBatchStride p.outputRowStride p.outputColumnStride
    p.startToken p.theta p.blockRows p.blockColumns

end
end VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch
