"""New spec-derived FP64 softmax sources (2026-10-24), not historical Python.

Correct.lean independently specifies exp(x[i])/sum_j exp(x[j]). The inputs
are loaded as FP32, then explicitly widened for FP64 reductions/arithmetic;
the output is explicitly narrowed back to FP32. The candidate changes only
division into reciprocal multiplication, retaining the existing rule/domain.
Triton v3.5.1 math.exp accepts FP64: https://github.com/triton-lang/triton/blob/v3.5.1/python/triton/language/math.py
"""
import triton
import triton.language as tl


@triton.jit
def floatStableSoftmaxKernel(xReg, yReg, blockSize: tl.constexpr):
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
def floatSoftmaxRecipKernel(xReg, yReg, blockSize: tl.constexpr):
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
