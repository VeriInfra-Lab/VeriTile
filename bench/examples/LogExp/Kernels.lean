import VeriTile.Triton.DSL
import VeriTile.Triton.DSL.CallSignature

/-! Spec-derived native sources are in native_kernels.py. These newly authored
sources do not claim a historical Python origin; see source-references.json. -/

namespace VeriTile.Bench.Examples.LogExp
open VeriTile Triton

def callSignature : CallSignature where
  pointers := [("xReg", .fp32), ("yReg", .fp32)]
  scalars := []
  constexpr := ["blockSize"]

def originalKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel :=
  with_triton_signature callSignature in
  triton {
    pid = tl.program_id(0)
    offs = pid * blockSize + tl.arange(0, blockSize)
    a = tl.load(xReg + offs)
    out = libdevice.log(libdevice.exp(a))
    tl.store(yReg + offs, out.to(tl.float32))
  }

def optimizedKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel :=
  with_triton_signature callSignature in
  triton {
    pid = tl.program_id(0)
    offs = pid * blockSize + tl.arange(0, blockSize)
    a = tl.load(xReg + offs)
    simplify = (0.5 < tl.abs(a)) & (tl.abs(a) <= 80.0)
    fallback_a = tl.where(simplify, 0.0, a)
    fallback = libdevice.log(libdevice.exp(fallback_a))
    out = tl.where(simplify, a, fallback)
    tl.store(yReg + offs, out.to(tl.float32))
  }

end VeriTile.Bench.Examples.LogExp
