"""New Python sources from the independent stable-softmax specification.

For one complete FP32 tile, compute exp(x-max(x))/sum(exp(x-max(x))) and
convert the output to BF16. The comparison replaces per-lane division with
multiplication by a shared reciprocal. Its FP proof still requires the stated
reciprocal rule and operand-domain guards. No historical Python provenance is
claimed, and tl.exp is retained in both sources.
"""
import triton
import triton.language as tl


@triton.jit
def stable_softmax(xReg, yReg, blockSize: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * blockSize + tl.arange(0, blockSize)
    x = tl.load(xReg + offs)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(yReg + offs, y.to(tl.bfloat16))


@triton.jit
def reciprocal_softmax(xReg, yReg, blockSize: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * blockSize + tl.arange(0, blockSize)
    x = tl.load(xReg + offs)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    inv_s = 1 / s
    y = e * inv_s
    tl.store(yReg + offs, y.to(tl.bfloat16))
