import VeriTile.Triton.DSL.CallSignature

/-! Newly authored Python sources from residual + silu(x * gate). The
FP64 input/scratch and BF16 output types are explicit invocation metadata.
The unfused program is the existing composition of the three actual JIT bodies. -/
namespace VeriTile.Bench.Examples.FusedSiLU.Kernels
open VeriTile Triton

def fusedSiLUKernelSignature : CallSignature where
  pointers := [("xReg",.fp64),("gateReg",.fp64),("residualReg",.fp64),("outReg",.bf16)]
  scalars := []
  constexpr := ["blockSize"]

def fusedSiLUKernel (xReg gateReg residualReg outReg : RegionName) (blockSize : Nat) : ComputeKernel :=
  with_triton_signature fusedSiLUKernelSignature in triton {
    pid      = tl.program_id(0)
    offsets  = pid * blockSize + tl.arange(0, blockSize)
    x        = tl.load(xReg + offsets)
    gate     = tl.load(gateReg + offsets)
    residual = tl.load(residualReg + offsets)
    z        = x * gate
    silu     = z * tl.sigmoid(z)
    y        = residual + silu
    tl.store(outReg + offsets, (y).to(tl.bfloat16))
  }

def siluStepGateSignature : CallSignature where
  pointers := [("xReg",.fp64),("gateReg",.fp64),("zReg",.fp64)]
  scalars := []
  constexpr := ["blockSize"]

def siluStepGateKernel (xReg gateReg zReg : RegionName) (blockSize : Nat) : ComputeKernel :=
  with_triton_signature siluStepGateSignature in triton {
    pid     = tl.program_id(0)
    offsets = pid * blockSize + tl.arange(0, blockSize)
    x       = tl.load(xReg + offsets)
    gate    = tl.load(gateReg + offsets)
    z       = x * gate
    tl.store(zReg + offsets, z)
  }

def siluStepSiluSignature : CallSignature where
  pointers := [("zReg",.fp64),("siluReg",.fp64)]
  scalars := []
  constexpr := ["blockSize"]

def siluStepSiluKernel (zReg siluReg : RegionName) (blockSize : Nat) : ComputeKernel :=
  with_triton_signature siluStepSiluSignature in triton {
    pid     = tl.program_id(0)
    offsets = pid * blockSize + tl.arange(0, blockSize)
    z       = tl.load(zReg + offsets)
    silu    = z * tl.sigmoid(z)
    tl.store(siluReg + offsets, silu)
  }

def siluStepResidualSignature : CallSignature where
  pointers := [("siluReg",.fp64),("residualReg",.fp64),("outReg",.bf16)]
  scalars := []
  constexpr := ["blockSize"]

def siluStepResidualKernel (siluReg residualReg outReg : RegionName) (blockSize : Nat) : ComputeKernel :=
  with_triton_signature siluStepResidualSignature in triton {
    pid      = tl.program_id(0)
    offsets  = pid * blockSize + tl.arange(0, blockSize)
    silu     = tl.load(siluReg + offsets)
    residual = tl.load(residualReg + offsets)
    y        = residual + silu
    tl.store(outReg + offsets, (y).to(tl.bfloat16))
  }

def unfusedSiLUKernel
    (xReg gateReg residualReg zReg siluReg outReg : RegionName)
    (blockSize : Nat) : ComputeKernel :=
  ComputeKernel.seq [xReg, gateReg, residualReg] [outReg]
    [siluStepGateKernel xReg gateReg zReg blockSize,
     siluStepSiluKernel zReg siluReg blockSize,
     siluStepResidualKernel siluReg residualReg outReg blockSize]

end VeriTile.Bench.Examples.FusedSiLU.Kernels
