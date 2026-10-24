"""RMSNorm for ordinary contiguous FP32 CUDA tensors.

This separate repair preserves the original arithmetic and default fusion.
The feature dimension remains limited to the original 64 KiB FP32 block.
"""

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rms_norm_fwd_fused(X, Y, W, stride, N, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    Y += row * stride
    X += row * stride
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = x * rstd
        y = x_hat * w
        tl.store(Y + cols, y, mask=mask)


class TritonLlamaRMSNorm(nn.Module):
    def __init__(self, weight, eps=1e-6):
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps

    def forward(self, x):
        weight = self.weight
        eps = self.variance_epsilon
        if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
            raise TypeError("x and weight must be tensors")
        if any(t.layout != torch.strided or t.dtype != torch.float32 or not t.is_cuda
               for t in (x, weight)):
            raise TypeError("x and weight must be strided FP32 CUDA tensors")
        if x.device != weight.device:
            raise ValueError("x and weight must be on the same device")
        if x.ndim < 1 or x.shape[-1] <= 0 or weight.shape != (x.shape[-1],):
            raise ValueError("x must have a positive feature dimension and weight must match it")
        if any(not t.is_contiguous() or t.storage_offset() != 0 or t.is_neg() or t.is_conj()
               for t in (x, weight)):
            raise ValueError("x and weight must be contiguous with zero offsets and no lazy flags")
        if type(eps) is not float or not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be a finite positive Python float")
        N = x.shape[-1]
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        if x.numel() > (2**63 - 1) // 4:
            raise ValueError("tensor byte extent must fit signed 64-bit addressing")
        if x.numel() // N > 2**31 - 1:
            raise ValueError("row count must fit the CUDA one-dimensional grid")
        if x.numel() != 0 and (x.data_ptr() % 4 != 0 or weight.data_ptr() % 4 != 0):
            raise ValueError("nonempty FP32 inputs must be 4-byte aligned")
        y = torch.empty_like(x, memory_format=torch.contiguous_format)
        x_arg = x.reshape(-1, N)
        M = x_arg.shape[0]
        if M == 0:
            return y
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        with torch.cuda.device(x.device):
            rms_norm_fwd_fused[(M,)](
                x_arg, y, weight, N, N, eps,
                BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
            )
        return y
