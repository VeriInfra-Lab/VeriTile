"""New Python worked-example sources from the independent row-softmax formula.

Input is a complete FP32 row, output is BF16. The reference normalizes exp(x);
the comparison subtracts the row maximum before exponentiation. Both use
libdevice.exp as required by the existing admitted FP rule. These are newly
authored sources, not recovered historical benchmark Python.
"""
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice


@triton.jit
def naive_softmax(xReg, yReg, B: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    x = tl.load(xReg + offs)
    e = libdevice.exp(x)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(yReg + offs, (y).to(tl.bfloat16))


@triton.jit
def stable_softmax(xReg, yReg, B: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    x = tl.load(xReg + offs)
    m = tl.max(x, axis=0)
    e = libdevice.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(yReg + offs, (y).to(tl.bfloat16))
