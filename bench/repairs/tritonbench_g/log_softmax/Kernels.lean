import VeriTile.Triton
import VeriTile.Triton.DSL.CallSignature

/-! Complete selected forward/backward bodies and existing address candidates.
Caller storage, runtime dimensions and constexpr tile sizes stay outside.
All source index casts, clamping, masks and floating operations are retained. -/
namespace VeriTile.Bench.Repairs.LogSoftmax.Source
open VeriTile.Triton

def forwardSignature : CallSignature where
  pointers := [("output_ptr",.fp32),("input_ptr",.fp32)]
  scalars := [("M",.integer),("N",.integer),("K",.integer)]
  constexpr := ["BLOCK_M","BLOCK_N"]

def forward (output_ptr input_ptr : RegionName) (M N K BM BN : Nat) : ComputeKernel :=
  let BLOCK_M := BM
  let BLOCK_N := BN
  with_triton_call forwardSignature call_args [M := M, N := N, K := K] in triton {
    pid_m = tl.program_id(0).to(tl.int64)
    pid_k = tl.program_id(1).to(tl.int64)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    n_offset = tl.arange(0, BLOCK_N).to(tl.int64)
    m64, n64, k64 = (tl.cast(M, tl.int64), tl.cast(N, tl.int64), tl.cast(K, tl.int64))
    safe_m = tl.minimum(m_offset, m64 - 1)
    safe_n = tl.minimum(n_offset, n64 - 1)
    safe_k = tl.minimum(pid_k, k64 - 1)
    offset = safe_m[:, None] * n64 * k64 + safe_n[None, :] * k64 + safe_k
    mask = (m_offset[:, None] < m64) & (n_offset[None, :] < n64) & (pid_k < k64)
    inp = tl.load(input_ptr + offset, mask=mask, other=-float('inf')).to(tl.float32)
    row_minus_max = inp - tl.max(inp, axis=1)[:, None]
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=1)[:, None]
    result = row_minus_max - tl.log(denominator)
    tl.store(output_ptr + offset, result, mask=mask)
  }

def forwardCommuted (output_ptr input_ptr : RegionName) (M N K BM BN : Nat) : ComputeKernel :=
  let BLOCK_M := BM
  let BLOCK_N := BN
  with_triton_call forwardSignature call_args [M := M, N := N, K := K] in triton {
    pid_m = tl.program_id(0).to(tl.int64)
    pid_k = tl.program_id(1).to(tl.int64)
    m_offset = BLOCK_M * pid_m + tl.arange(0, BLOCK_M).to(tl.int64)
    n_offset = tl.arange(0, BLOCK_N).to(tl.int64)
    m64, n64, k64 = (tl.cast(M, tl.int64), tl.cast(N, tl.int64), tl.cast(K, tl.int64))
    safe_m = tl.minimum(m_offset, m64 - 1)
    safe_n = tl.minimum(n_offset, n64 - 1)
    safe_k = tl.minimum(pid_k, k64 - 1)
    offset = safe_m[:, None] * n64 * k64 + safe_n[None, :] * k64 + safe_k
    mask = (m_offset[:, None] < m64) & (n_offset[None, :] < n64) & (pid_k < k64)
    inp = tl.load(input_ptr + offset, mask=mask, other=-float('inf')).to(tl.float32)
    row_minus_max = inp - tl.max(inp, axis=1)[:, None]
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=1)[:, None]
    result = row_minus_max - tl.log(denominator)
    tl.store(output_ptr + offset, result, mask=mask)
  }

def backwardSignature : CallSignature where
  pointers := [("out_ptr",.fp32),("out_grad_ptr",.fp32),("in_grad_ptr",.fp32)]
  scalars := [("M",.integer),("N",.integer),("K",.integer)]
  constexpr := ["BLOCK_M","BLOCK_N"]

def backward (out_ptr out_grad_ptr in_grad_ptr : RegionName) (M N K BM BN : Nat) : ComputeKernel :=
  let BLOCK_M := BM
  let BLOCK_N := BN
  with_triton_call backwardSignature call_args [M := M, N := N, K := K] in triton {
    pid_m = tl.program_id(0).to(tl.int64)
    pid_k = tl.program_id(1).to(tl.int64)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    n_offset = tl.arange(0, BLOCK_N).to(tl.int64)
    m64, n64, k64 = (tl.cast(M, tl.int64), tl.cast(N, tl.int64), tl.cast(K, tl.int64))
    safe_m = tl.minimum(m_offset, m64 - 1)
    safe_n = tl.minimum(n_offset, n64 - 1)
    safe_k = tl.minimum(pid_k, k64 - 1)
    offset = safe_m[:, None] * n64 * k64 + safe_n[None, :] * k64 + safe_k
    mask = (m_offset[:, None] < m64) & (n_offset[None, :] < n64) & (pid_k < k64)
    out = tl.load(out_ptr + offset, mask=mask, other=-float('inf')).to(tl.float32)
    out_grad = tl.load(out_grad_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    scale = tl.sum(out_grad, axis=1)
    in_grad = out_grad - tl.exp(out.to(tl.float32)) * scale[:, None]
    tl.store(in_grad_ptr + offset, in_grad, mask=mask)
  }

def backwardCommuted (out_ptr out_grad_ptr in_grad_ptr : RegionName) (M N K BM BN : Nat) : ComputeKernel :=
  let BLOCK_M := BM
  let BLOCK_N := BN
  with_triton_call backwardSignature call_args [M := M, N := N, K := K] in triton {
    pid_m = tl.program_id(0).to(tl.int64)
    pid_k = tl.program_id(1).to(tl.int64)
    m_offset = BLOCK_M * pid_m + tl.arange(0, BLOCK_M).to(tl.int64)
    n_offset = tl.arange(0, BLOCK_N).to(tl.int64)
    m64, n64, k64 = (tl.cast(M, tl.int64), tl.cast(N, tl.int64), tl.cast(K, tl.int64))
    safe_m = tl.minimum(m_offset, m64 - 1)
    safe_n = tl.minimum(n_offset, n64 - 1)
    safe_k = tl.minimum(pid_k, k64 - 1)
    offset = safe_m[:, None] * n64 * k64 + safe_n[None, :] * k64 + safe_k
    mask = (m_offset[:, None] < m64) & (n_offset[None, :] < n64) & (pid_k < k64)
    out = tl.load(out_ptr + offset, mask=mask, other=-float('inf')).to(tl.float32)
    out_grad = tl.load(out_grad_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    scale = tl.sum(out_grad, axis=1)
    in_grad = out_grad - tl.exp(out.to(tl.float32)) * scale[:, None]
    tl.store(in_grad_ptr + offset, in_grad, mask=mask)
  }

end VeriTile.Bench.Repairs.LogSoftmax.Source
