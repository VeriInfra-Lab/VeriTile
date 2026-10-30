"""Spec-derived native softmax sources, authored 2026-10-24.
Not recovered historical Python. Mathematical target: exp(x_i)/sum exp(x_j),
as stated independently in Correct.lean. The existing max-shift transform,
BF16 output and admitted libdevice.exp numerical domain remain explicit.
ABI: FP32 xReg, BF16 yReg, positive constexpr B, complete tiles.
"""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def naiveSoftmaxKernel(xReg, yReg, B: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    x = tl.load(xReg + offs)
    e = libdevice.exp(x)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(yReg + offs, y.to(tl.bfloat16))


@triton.jit
def stableSoftmaxKernel(xReg, yReg, B: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    x = tl.load(xReg + offs)
    m = tl.max(x, axis=0)
    e = libdevice.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(yReg + offs, y.to(tl.bfloat16))
