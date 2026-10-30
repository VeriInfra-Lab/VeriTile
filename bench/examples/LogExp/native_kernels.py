"""Spec-derived native source, authored 2026-10-24 (not a historical benchmark).

Independent mathematical target: y[i] = x[i], see Correct.lean's
original_correct and optimized_correct. The reference evaluates log(exp(x));
the existing candidate uses identity on 0.5 < abs(x) <= 80 and the reference
otherwise. FP equivalence retains the separate conditional LogExp rules.
ABI: FP32 pointers; blockSize is constexpr. Launch covers complete tiles.
"""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def originalKernel(xReg, yReg, blockSize: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * blockSize + tl.arange(0, blockSize)
    a = tl.load(xReg + offs)
    out = libdevice.log(libdevice.exp(a))
    tl.store(yReg + offs, out.to(tl.float32))


@triton.jit
def optimizedKernel(xReg, yReg, blockSize: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * blockSize + tl.arange(0, blockSize)
    a = tl.load(xReg + offs)
    simplify = (0.5 < tl.abs(a)) & (tl.abs(a) <= 80.0)
    fallback_a = tl.where(simplify, 0.0, a)
    fallback = libdevice.log(libdevice.exp(fallback_a))
    out = tl.where(simplify, a, fallback)
    tl.store(yReg + offs, out.to(tl.float32))
