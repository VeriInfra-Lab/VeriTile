import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Source

/-! Python sources newly authored from the independent mathematical
specification; see softmax_reciprocal.py for provenance. Intrinsics, guards, arithmetic
order and explicit output conversions are retained in the pasted bodies. -/
namespace VeriTile.Bench.Examples.SoftmaxReciprocal.Kernels
open VeriTile Triton

def callSignature : CallSignature where
  pointers := [("xReg",.fp32),("yReg",.bf16)]
  scalars := []
  constexpr := ["blockSize"]

def stable_softmaxKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel :=
  with_triton_signature callSignature in triton {
    pid = tl.program_id(0)
    offs = pid * blockSize + tl.arange(0, blockSize)
    x = tl.load(xReg + offs)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(yReg + offs, y.to(tl.bfloat16))
  }

def reciprocal_softmaxKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel :=
  with_triton_signature callSignature in triton {
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

/-- Domain prefix derived from the actual original program, before y. -/
def commonPrefix (xReg : RegionName) (B : Nat) : ComputeKernel :=
  let selected := stable_softmaxKernel xReg "__unused_output" B
  .mk [xReg] [] (Source.prefixBefore selected.surfaceBody "y")

def originalKernel (xReg yReg : RegionName) (B : Nat) : ComputeKernel :=
  stable_softmaxKernel xReg yReg B

def reciprocalKernel (xReg yReg : RegionName) (B : Nat) : ComputeKernel :=
  reciprocal_softmaxKernel xReg yReg B

end VeriTile.Bench.Examples.SoftmaxReciprocal.Kernels
