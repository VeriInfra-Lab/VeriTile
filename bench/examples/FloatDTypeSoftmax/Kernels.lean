import VeriTile.Triton.DSL
import VeriTile.Triton.DSL.CallSignature

/-! New spec-derived Python sources; see native_kernels.py and source-references.json.
FP32 load, FP64 arithmetic and FP32 store boundaries are retained. -/

namespace VeriTile.Bench.Examples.FloatDTypeSoftmax.Kernels
open VeriTile Triton

def callSignature : CallSignature where
  pointers := [("xReg",.fp32),("yReg",.fp32)]
  scalars := []
  constexpr := ["blockSize"]

def floatStableSoftmaxKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel :=
  with_triton_signature callSignature in
  triton {
    pid = tl.program_id(0)
    offs = pid * blockSize + tl.arange(0, blockSize)
    x32 = tl.load(xReg + offs)
    x = x32.to(tl.float64)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(yReg + offs, y.to(tl.float32))
  }

def floatSoftmaxRecipKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel :=
  with_triton_signature callSignature in
  triton {
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
  }

def originalKernel (xReg yReg : RegionName) (B : Nat) : ComputeKernel :=
  floatStableSoftmaxKernel xReg yReg B

def reciprocalKernel (xReg yReg : RegionName) (B : Nat) : ComputeKernel :=
  floatSoftmaxRecipKernel xReg yReg B

end VeriTile.Bench.Examples.FloatDTypeSoftmax.Kernels
