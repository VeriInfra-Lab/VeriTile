import VeriTile.Triton.DSL
import VeriTile.Triton.DSL.CallSignature

/-! Spec-derived native sources are in native_kernels.py. These newly authored
sources do not claim a historical Python origin; see source-references.json. -/

namespace VeriTile.Bench.Examples.SoftmaxReciprocal.Kernels
open VeriTile Triton

def callSignature : CallSignature where
  pointers := [("xReg", .fp32), ("yReg", .bf16)]
  scalars := []
  constexpr := ["blockSize"]

def stableSoftmaxKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel :=
  with_triton_signature callSignature in
  triton {
    pid = tl.program_id(0)
    offs = pid * blockSize + tl.arange(0, blockSize)
    x = tl.load(xReg + offs)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(yReg + offs, y.to(tl.bfloat16))
  }

def softmaxRecipKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel :=
  with_triton_signature callSignature in
  triton {
    pid = tl.program_id(0)
    offs = pid * blockSize + tl.arange(0, blockSize)
    x = tl.load(xReg + offs)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    inv_s = 1 / s
    y = e * inv_s
    tl.store(yReg + offs, y.to(tl.bfloat16))
  }

def originalKernel (xReg yReg : RegionName) (B : Nat) : ComputeKernel :=
  stableSoftmaxKernel xReg yReg B

def reciprocalKernel (xReg yReg : RegionName) (B : Nat) : ComputeKernel :=
  softmaxRecipKernel xReg yReg B

end VeriTile.Bench.Examples.SoftmaxReciprocal.Kernels
