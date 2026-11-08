"""New Python sources derived from the independent identity specification.

The reference computes libdevice.log(libdevice.exp(a)). The candidate returns
a only in its declared numerical domain and keeps the original expression
elsewhere. These are newly authored worked-example sources; no recovered
historical Python implementation is claimed. Both libdevice intrinsics and
the explicit FP32 output conversion are retained.
"""
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice


@triton.jit
def log_exp(xReg, yReg, blockSize: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * blockSize + tl.arange(0, blockSize)
    a = tl.load(xReg + offs)
    out = libdevice.log(libdevice.exp(a))
    tl.store(yReg + offs, out.to(tl.float32))


@triton.jit
def log_exp_guarded(xReg, yReg, blockSize: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * blockSize + tl.arange(0, blockSize)
    a = tl.load(xReg + offs)
    simplify = (0.5 < tl.abs(a)) & (tl.abs(a) <= 80.0)
    fallback_a = tl.where(simplify, 0.0, a)
    fallback = libdevice.log(libdevice.exp(fallback_a))
    out = tl.where(simplify, a, fallback)
    tl.store(yReg + offs, out.to(tl.float32))
