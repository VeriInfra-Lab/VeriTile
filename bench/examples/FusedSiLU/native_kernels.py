"""New Python from the independent fused SiLU mathematical specification.
No historical Python source is claimed. FP32 inputs and FP32 intermediate
scratch preserve each separate operation; the final output is BF16. Use
three ordered launches for the unfused stages and enable_fp_fusion=False.
The proofs use the existing scalarFP32 software word interpretation; sigmoid
and mixed-format conversion stay explicit opaque primitives. This is not a
GPU or compiler conformance claim.
"""
import triton
import triton.language as tl

@triton.jit
def fusedSiLUKernel(xReg, gateReg, residualReg, outReg, blockSize: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * blockSize + tl.arange(0, blockSize)
    x = tl.load(xReg + offsets)
    gate = tl.load(gateReg + offsets)
    residual = tl.load(residualReg + offsets)
    z = x * gate
    silu = z * tl.sigmoid(z)
    y = residual + silu
    tl.store(outReg + offsets, y.to(tl.bfloat16))

@triton.jit
def siluStepGate(xReg, gateReg, zReg, blockSize: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * blockSize + tl.arange(0, blockSize)
    x = tl.load(xReg + offsets)
    gate = tl.load(gateReg + offsets)
    z = x * gate
    tl.store(zReg + offsets, z)

@triton.jit
def siluStepSilu(zReg, siluReg, blockSize: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * blockSize + tl.arange(0, blockSize)
    z = tl.load(zReg + offsets)
    silu = z * tl.sigmoid(z)
    tl.store(siluReg + offsets, silu)

@triton.jit
def siluStepResidual(siluReg, residualReg, outReg, blockSize: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * blockSize + tl.arange(0, blockSize)
    silu = tl.load(siluReg + offsets)
    residual = tl.load(residualReg + offsets)
    y = residual + silu
    tl.store(outReg + offsets, y.to(tl.bfloat16))
