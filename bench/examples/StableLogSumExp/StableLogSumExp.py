"""New Python sources from the independent log(sum(exp(x))) specification.

Input storage is FP32, output is BF16. The retained comparison uses the
previously specified guarded log-product splitting and log-exp cancellation,
including both numerical fallback branches. These are newly authored worked
examples; no historical Python provenance is claimed.
"""
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice


@triton.jit
def direct_log_sum_exp(xReg, yReg, blockSize: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * blockSize + tl.arange(0, blockSize)
    x    = tl.load(xReg + offs)
    e    = libdevice.exp(x)
    s    = tl.sum(e, axis=0)
    y    = tl.log(s)
    tl.store(yReg + pid, (y).to(tl.bfloat16))


@triton.jit
def guarded_log_sum_exp(xReg, yReg, blockSize: tl.constexpr):
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
    tl.store(yReg + pid, (y).to(tl.bfloat16))
