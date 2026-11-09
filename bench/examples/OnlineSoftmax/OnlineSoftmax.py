"""New native sources from the independent softmax specification.
No historical Python origin is claimed. Inputs/outputs are FP32; N is the
contiguous per-program row width. Online softmax retains both input passes
and every output store. The typed scalar initializers keep loop state FP32.
The source uses libdevice.exp, matching the existing admitted exp-sub rules.
Use enable_fp_fusion=False for the separate arithmetic graph represented by
the proofs. No GPU or compiler conformance experiment is claimed here.
"""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def stableSoftmaxKernel(xReg, yReg, blockSize: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * blockSize + tl.arange(0, blockSize)
    x = tl.load(xReg + offs)
    m = tl.max(x, axis=0)
    e = libdevice.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(yReg + offs, y)


@triton.jit
def onlineSoftmaxKernel(xReg, yReg, N: tl.constexpr):
    pid = tl.program_id(0)
    m = tl.full((), float("-inf"), tl.float32)
    l = tl.full((), 0, tl.float32)
    for i in range(N):
        xi = tl.load(xReg + (pid * N + i))
        m_new = tl.maximum(m, xi)
        l = libdevice.exp(m - m_new) * l + libdevice.exp(xi - m_new)
        m = m_new
    pid = tl.program_id(0)
    offs = pid * N + tl.arange(0, N)
    x = tl.load(xReg + offs)
    e = libdevice.exp(x - m)
    y = e / l
    tl.store(yReg + offs, y)
