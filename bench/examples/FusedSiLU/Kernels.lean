import VeriTile.Triton.DSL.CallSignature

/-! Newly authored native source from the independent fused SiLU formula;
no historical Python origin is asserted. FP32 scratch is materialized and
reloaded in the three-stage candidate; every final store converts to BF16.
Separate arithmetic uses enable_fp_fusion=False. -/

namespace VeriTile.Bench.Examples.FusedSiLU.Kernels
open VeriTile Triton

def fusedSiLUKernelSignature : CallSignature where
  pointers := [("xReg",.fp32), ("gateReg",.fp32), ("residualReg",.fp32), ("outReg",.bf16)]
  scalars := []
  constexpr := ["blockSize"]

def fusedSiLUKernel (xReg gateReg residualReg outReg : RegionName) (blockSize : Nat) : ComputeKernel :=
  with_triton_signature fusedSiLUKernelSignature in triton {
    pid = tl.program_id(0)
    offsets = pid * blockSize + tl.arange(0, blockSize)
    x = tl.load(xReg + offsets)
    gate = tl.load(gateReg + offsets)
    residual = tl.load(residualReg + offsets)
    z = x * gate
    silu = z * tl.sigmoid(z)
    y = residual + silu
    tl.store(outReg + offsets, y.to(tl.bfloat16))
  }

def siluStepGateSignature : CallSignature where
  pointers := [("xReg",.fp32), ("gateReg",.fp32), ("zReg",.fp32)]
  scalars := []
  constexpr := ["blockSize"]

def siluStepGate (xReg gateReg zReg : RegionName) (blockSize : Nat) : ComputeKernel :=
  with_triton_signature siluStepGateSignature in triton {
    pid = tl.program_id(0)
    offsets = pid * blockSize + tl.arange(0, blockSize)
    x = tl.load(xReg + offsets)
    gate = tl.load(gateReg + offsets)
    z = x * gate
    tl.store(zReg + offsets, z)
  }

def siluStepSiluSignature : CallSignature where
  pointers := [("zReg",.fp32), ("siluReg",.fp32)]
  scalars := []
  constexpr := ["blockSize"]

def siluStepSilu (zReg siluReg : RegionName) (blockSize : Nat) : ComputeKernel :=
  with_triton_signature siluStepSiluSignature in triton {
    pid = tl.program_id(0)
    offsets = pid * blockSize + tl.arange(0, blockSize)
    z = tl.load(zReg + offsets)
    silu = z * tl.sigmoid(z)
    tl.store(siluReg + offsets, silu)
  }

def siluStepResidualSignature : CallSignature where
  pointers := [("siluReg",.fp32), ("residualReg",.fp32), ("outReg",.bf16)]
  scalars := []
  constexpr := ["blockSize"]

def siluStepResidual (siluReg residualReg outReg : RegionName) (blockSize : Nat) : ComputeKernel :=
  with_triton_signature siluStepResidualSignature in triton {
    pid = tl.program_id(0)
    offsets = pid * blockSize + tl.arange(0, blockSize)
    silu = tl.load(siluReg + offsets)
    residual = tl.load(residualReg + offsets)
    y = residual + silu
    tl.store(outReg + offsets, y.to(tl.bfloat16))
  }

def unfusedSiLUKernel (xReg gateReg residualReg zReg siluReg outReg : RegionName) (blockSize : Nat) : ComputeKernel :=
  ComputeKernel.seq [xReg, gateReg, residualReg] [outReg]
    [siluStepGate xReg gateReg zReg blockSize, siluStepSilu zReg siluReg blockSize,
     siluStepResidual siluReg residualReg outReg blockSize]
end VeriTile.Bench.Examples.FusedSiLU.Kernels
