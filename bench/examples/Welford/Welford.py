"""New Python from the independent population mean/variance specification.
No recovered historical Python source is claimed. Both variants consume one
FP32 row of blockSize entries at pid*rowStride and write two BF16 scalar
outputs at address zero. This is a per-program contract, not a multi-row
output grid. The explicit integer-to-FP32 count conversions retain the
bounded count rules used by the proof. Use enable_fp_fusion=False for the
separate arithmetic graph represented here. This file contains source only;
no GPU execution or compiler conformance result is claimed.
"""
import triton
import triton.language as tl


@triton.jit
def twopassWelfordKernel(xReg, meanReg, varReg, blockSize: tl.constexpr, rowStride):
    pid = tl.program_id(0)
    offs = pid * rowStride + tl.arange(0, blockSize)
    x = tl.load(xReg + offs)
    count = tl.full((), blockSize, tl.uint64).to(tl.float32)
    s_x = tl.sum(x, axis=0)
    μ = s_x / count
    d = x - μ
    s_d2 = tl.sum(d * d, axis=0)
    v = s_d2 / count
    tl.store(meanReg, μ.to(tl.bfloat16))
    tl.store(varReg, v.to(tl.bfloat16))


@triton.jit
def onlineWelfordKernel(xReg, meanReg, varReg, blockSize: tl.constexpr, rowStride):
    pid = tl.program_id(0)
    M = 0.0
    S = 0.0
    for i in range(blockSize):
        xi = tl.load(xReg + (pid * rowStride + i))
        delta = xi - M
        M = M + delta / (i.to(tl.float32) + 1)
        delta2 = xi - M
        S = S + delta * delta2
    count = tl.full((), blockSize, tl.uint64).to(tl.float32)
    tl.store(meanReg, M.to(tl.bfloat16))
    tl.store(varReg, (S / count).to(tl.bfloat16))
