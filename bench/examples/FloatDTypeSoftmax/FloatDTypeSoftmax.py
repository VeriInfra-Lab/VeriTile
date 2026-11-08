"""New Python sources from the independent stable-softmax specification.

Input/output storage is FP32 and working arithmetic is FP64. The separate
FP32 load and FP64 conversion are intentional. The candidate uses a shared
reciprocal under the existing rule and intermediate-domain contract. These
are newly authored sources, not recovered historical benchmark code.
"""
import triton
import triton.language as tl


@triton.jit
def stable_softmax_fp64(xReg, yReg, blockSize: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * blockSize + tl.arange(0, blockSize)
    x32 = tl.load(xReg + offs)
    x = x32.to(tl.float64)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(yReg + offs, y.to(tl.float32))


@triton.jit
def reciprocal_softmax_fp64(xReg, yReg, blockSize: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * blockSize + tl.arange(0, blockSize)
    x32 = tl.load(xReg + offs)
    x = x32.to(tl.float64)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    inv_s = 1 / s
    y = e * inv_s
    tl.store(yReg + offs, y.to(tl.float32))
