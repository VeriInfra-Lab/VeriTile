import VeriTile.Triton

/-! Canonical FP32 repaired sources. Python explicitly widens each index to
signed int64; Proofs/Machine proves the correspondence with these exact Nat
indices, including inactive clamped lanes. The retained candidate commutes
only the row-index multiplication. No floating expression is reassociated. -/
namespace VeriTile.Bench.Repairs.LogSoftmax.Source
open VeriTile.Triton

def forward (output_ptr input_ptr : RegionName) (M N K BM BN : Nat) : ComputeKernel :=
  triton {
    tl.ports inputs=[input_ptr], outputs=[output_ptr]
    tl.region input_ptr = tl.float32, output_ptr = tl.float32
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_offset = pid_m * $(BM) + tl.arange(0, $(BM))
    n_offset = tl.arange(0, $(BN))
    safe_m = tl.minimum(m_offset, $(M - 1))
    safe_n = tl.minimum(n_offset, $(N - 1))
    safe_k = tl.minimum(pid_k, $(K - 1))
    offset = safe_m[:, None] * $(N) * $(K) + safe_n[None, :] * $(K) + safe_k
    mask = (m_offset[:, None] < $(M)) & (n_offset[None, :] < $(N)) & (pid_k < $(K))
    inp = tl.load(input_ptr + offset, mask=mask, other=-float("inf")).to(tl.float32)
    row_minus_max = inp - tl.max(inp, axis=1)[:, None]
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=1)[:, None]
    result = row_minus_max - tl.log(denominator)
    tl.store(output_ptr + offset, result, mask=mask)
  }


def forwardCommuted (output_ptr input_ptr : RegionName) (M N K BM BN : Nat) : ComputeKernel :=
  triton {
    tl.ports inputs=[input_ptr], outputs=[output_ptr]
    tl.region input_ptr = tl.float32, output_ptr = tl.float32
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_offset = $(BM) * pid_m + tl.arange(0, $(BM))
    n_offset = tl.arange(0, $(BN))
    safe_m = tl.minimum(m_offset, $(M - 1))
    safe_n = tl.minimum(n_offset, $(N - 1))
    safe_k = tl.minimum(pid_k, $(K - 1))
    offset = safe_m[:, None] * $(N) * $(K) + safe_n[None, :] * $(K) + safe_k
    mask = (m_offset[:, None] < $(M)) & (n_offset[None, :] < $(N)) & (pid_k < $(K))
    inp = tl.load(input_ptr + offset, mask=mask, other=-float("inf")).to(tl.float32)
    row_minus_max = inp - tl.max(inp, axis=1)[:, None]
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=1)[:, None]
    result = row_minus_max - tl.log(denominator)
    tl.store(output_ptr + offset, result, mask=mask)
  }


def backward (out_ptr out_grad_ptr in_grad_ptr : RegionName) (M N K BM BN : Nat) : ComputeKernel :=
  triton {
    tl.ports inputs=[out_ptr, out_grad_ptr], outputs=[in_grad_ptr]
    tl.region out_ptr = tl.float32, out_grad_ptr = tl.float32, in_grad_ptr = tl.float32
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_offset = pid_m * $(BM) + tl.arange(0, $(BM))
    n_offset = tl.arange(0, $(BN))
    safe_m = tl.minimum(m_offset, $(M - 1))
    safe_n = tl.minimum(n_offset, $(N - 1))
    safe_k = tl.minimum(pid_k, $(K - 1))
    offset = safe_m[:, None] * $(N) * $(K) + safe_n[None, :] * $(K) + safe_k
    mask = (m_offset[:, None] < $(M)) & (n_offset[None, :] < $(N)) & (pid_k < $(K))
    out = tl.load(out_ptr + offset, mask=mask, other=-float("inf")).to(tl.float32)
    out_grad = tl.load(out_grad_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    scale = tl.sum(out_grad, axis=1)
    in_grad = out_grad - tl.exp(out.to(tl.float32)) * scale[:, None]
    tl.store(in_grad_ptr + offset, in_grad, mask=mask)
  }


def backwardCommuted (out_ptr out_grad_ptr in_grad_ptr : RegionName) (M N K BM BN : Nat) : ComputeKernel :=
  triton {
    tl.ports inputs=[out_ptr, out_grad_ptr], outputs=[in_grad_ptr]
    tl.region out_ptr = tl.float32, out_grad_ptr = tl.float32, in_grad_ptr = tl.float32
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_offset = $(BM) * pid_m + tl.arange(0, $(BM))
    n_offset = tl.arange(0, $(BN))
    safe_m = tl.minimum(m_offset, $(M - 1))
    safe_n = tl.minimum(n_offset, $(N - 1))
    safe_k = tl.minimum(pid_k, $(K - 1))
    offset = safe_m[:, None] * $(N) * $(K) + safe_n[None, :] * $(K) + safe_k
    mask = (m_offset[:, None] < $(M)) & (n_offset[None, :] < $(N)) & (pid_k < $(K))
    out = tl.load(out_ptr + offset, mask=mask, other=-float("inf")).to(tl.float32)
    out_grad = tl.load(out_grad_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    scale = tl.sum(out_grad, axis=1)
    in_grad = out_grad - tl.exp(out.to(tl.float32)) * scale[:, None]
    tl.store(in_grad_ptr + offset, in_grad, mask=mask)
  }

end VeriTile.Bench.Repairs.LogSoftmax.Source
