"""Spec-derived native source, authored 2026-10-24 (not a historical benchmark).

Independent mathematical target: exp(x[i]) / sum_j exp(x[j]), using a common
maximum shift; see Correct.lean's public specifications. The existing
candidate computes a reciprocal once. FP equivalence retains div_mul_rcp
and its finite/nonzero domain, with no exp/libdevice interchange.
ABI: FP32 xReg, BF16 yReg; blockSize is constexpr, positive, a complete tile.
"""
import triton
import triton.language as tl


@triton.jit
def stableSoftmaxKernel(xReg, yReg, blockSize: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * blockSize + tl.arange(0, blockSize)
    x = tl.load(xReg + offs)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(yReg + offs, y.to(tl.bfloat16))


@triton.jit
def softmaxRecipKernel(xReg, yReg, blockSize: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * blockSize + tl.arange(0, blockSize)
    x = tl.load(xReg + offs)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    inv_s = 1 / s
    y = e * inv_s
    tl.store(yReg + offs, y.to(tl.bfloat16))
