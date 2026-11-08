import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Source

/-! Newly authored Python sources from the independent stable-softmax
specification; see float_dtype_softmax.py. Both precision boundaries and the
reciprocal transformation remain explicit in the copied function bodies. -/
namespace VeriTile.Bench.Examples.FloatDTypeSoftmax.Kernels
open VeriTile Triton

def callSignature : CallSignature where
  pointers := [("xReg",.fp32),("yReg",.fp32)]
  scalars := []
  constexpr := ["blockSize"]

def floatStableSoftmaxKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel :=
  with_triton_signature callSignature in triton {
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
  with_triton_signature callSignature in triton {
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

/-- The actual original prefix, preserving x32 and its separate fp64 cast. -/
def commonPrefix (xReg : RegionName) (B : Nat) : ComputeKernel :=
  let selected := floatStableSoftmaxKernel xReg "__unused_output" B
  .mk [xReg] [] (Source.prefixBefore selected.surfaceBody "y")

def originalKernel (xReg yReg : RegionName) (B : Nat) : ComputeKernel :=
  floatStableSoftmaxKernel xReg yReg B

def reciprocalKernel (xReg yReg : RegionName) (B : Nat) : ComputeKernel :=
  floatSoftmaxRecipKernel xReg yReg B

end VeriTile.Bench.Examples.FloatDTypeSoftmax.Kernels
