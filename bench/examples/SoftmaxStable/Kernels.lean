import VeriTile.Triton.DSL
import VeriTile.Triton.DSL.CallSignature

/-! New spec-derived native Python, not historical source; see source-references.json.
The intrinsic choices, conditional fallbacks and BF16 stores are explicit. -/

namespace VeriTile.Bench.Examples.SoftmaxStable.Kernels
open VeriTile Triton

def callSignature : CallSignature where
  pointers := [("xReg",.fp32),("yReg",.bf16)]
  scalars := []
  constexpr := ["B"]

def naiveSoftmaxKernel (xReg yReg : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature callSignature in
  triton {
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    x = tl.load(xReg + offs)
    e = libdevice.exp(x)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(yReg + offs, y.to(tl.bfloat16))
  }

def stableSoftmaxKernel (xReg yReg : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature callSignature in
  triton {
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    x = tl.load(xReg + offs)
    m = tl.max(x, axis=0)
    e = libdevice.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(yReg + offs, y.to(tl.bfloat16))
  }

end VeriTile.Bench.Examples.SoftmaxStable.Kernels
