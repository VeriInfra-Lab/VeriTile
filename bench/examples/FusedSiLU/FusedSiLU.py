"""New Python sources from residual + silu(x * gate).

The independent mathematical specification determines both the fused result
and the three-stage decomposition. FP64 input and scratch storage preserve
this worked example's real-carrier intermediates; the final result uses BF16
storage. These newly authored examples are not recovered benchmark sources.
"""
import triton
import triton.language as tl

@triton.jit
def fused_silu(xReg, gateReg, residualReg, outReg, blockSize: tl.constexpr):
    pid      = tl.program_id(0)
    offsets  = pid * blockSize + tl.arange(0, blockSize)
    x        = tl.load(xReg + offsets)
    gate     = tl.load(gateReg + offsets)
    residual = tl.load(residualReg + offsets)
    z        = x * gate
    silu     = z * tl.sigmoid(z)
    y        = residual + silu
    tl.store(outReg + offsets, (y).to(tl.bfloat16))


@triton.jit
def silu_gate(xReg, gateReg, zReg, blockSize: tl.constexpr):
    pid     = tl.program_id(0)
    offsets = pid * blockSize + tl.arange(0, blockSize)
    x       = tl.load(xReg + offsets)
    gate    = tl.load(gateReg + offsets)
    z       = x * gate
    tl.store(zReg + offsets, z)


@triton.jit
def silu_activation(zReg, siluReg, blockSize: tl.constexpr):
    pid     = tl.program_id(0)
    offsets = pid * blockSize + tl.arange(0, blockSize)
    z       = tl.load(zReg + offsets)
    silu    = z * tl.sigmoid(z)
    tl.store(siluReg + offsets, silu)


@triton.jit
def silu_residual(siluReg, residualReg, outReg, blockSize: tl.constexpr):
    pid      = tl.program_id(0)
    offsets  = pid * blockSize + tl.arange(0, blockSize)
    silu     = tl.load(siluReg + offsets)
    residual = tl.load(residualReg + offsets)
    y        = residual + silu
    tl.store(outReg + offsets, (y).to(tl.bfloat16))
