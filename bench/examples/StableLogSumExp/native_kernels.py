"""Spec-derived native log-sum-exp sources, authored 2026-10-24.
Not historical Python. Independent target: log(sum exp(x_i)), Correct.lean.
The prior shifted reference and existing conditional candidate are retained.
All logs remain tl.log, exponentials libdevice.exp; masked inactive arguments
and BF16 output conversions remain explicit. FP conditions are unchanged in
kind and must apply to the actual FP32-loaded values and reduction schedule.
ABI: FP32 xReg, BF16 yReg; positive constexpr blockSize, complete tiles.
"""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def directLSEKernel(xReg, yReg, blockSize: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * blockSize + tl.arange(0, blockSize)
    x = tl.load(xReg + offs)
    e = libdevice.exp(x)
    s = tl.sum(e, axis=0)
    y = tl.log(s)
    tl.store(yReg + pid, y.to(tl.bfloat16))


@triton.jit
def stableLSEKernel(xReg, yReg, blockSize: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * blockSize + tl.arange(0, blockSize)
    x = tl.load(xReg + offs)
    m = tl.max(x, axis=0)
    e = libdevice.exp(x - m)
    s = tl.sum(e, axis=0)
    y = m + tl.log(s)
    tl.store(yReg + pid, y.to(tl.bfloat16))


@triton.jit
def optimizedLSEKernel(xReg, yReg, blockSize: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * blockSize + tl.arange(0, blockSize)
    x = tl.load(xReg + offs)
    m = tl.max(x, axis=0)
    e = libdevice.exp(x - m)
    s = tl.sum(e, axis=0)
    p = s * libdevice.exp(m)
    keep = (p >= 0.5) & (p <= 2.0)
    direct = tl.log(tl.where(keep, p, 1.0))
    split_m = tl.where(keep, 0.0, m)
    simplify = (0.5 < tl.abs(split_m)) & (tl.abs(split_m) <= 80.0)
    fallback = tl.log(libdevice.exp(tl.where(simplify, 0.0, split_m)))
    log_em = tl.where(simplify, split_m, fallback)
    split = tl.log(tl.where(keep, 1.0, s)) + log_em
    y = tl.where(keep, direct, split)
    tl.store(yReg + pid, y.to(tl.bfloat16))
