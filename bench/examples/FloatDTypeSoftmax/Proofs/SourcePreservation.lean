import bench.examples.FloatDTypeSoftmax.Kernels
import bench.examples.FloatDTypeSoftmax.Proofs.LegacySurface

namespace VeriTile.Bench.Examples.FloatDTypeSoftmax.Kernels
open VeriTile Triton

theorem floatStableSoftmaxKernel_preserved (xReg yReg : RegionName) (B : Nat) :
    floatStableSoftmaxKernel xReg yReg B = LegacySurface.floatStableSoftmaxKernel xReg yReg B := rfl

theorem floatSoftmaxRecipKernel_preserved (xReg yReg : RegionName) (B : Nat) :
    floatSoftmaxRecipKernel xReg yReg B = LegacySurface.floatSoftmaxRecipKernel xReg yReg B := rfl

/-- Proof-only common prefix; connected to the full native source. -/
def commonPrefix (xReg : RegionName) (B : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(B) + tl.arange(0, $(B))
  x32  := tl.load($(xReg) + offs, dtype=tl.float32)
  x    := x32.to(tl.float64)
  m    := tl.max(x, axis=0)
  e    := tl.exp(x - m)
  s    := tl.sum(e, axis=0)
}

theorem commonPrefix_source (xReg yReg : RegionName) (B : Nat) :
    (commonPrefix xReg B).surfaceBody =
      (floatStableSoftmaxKernel xReg yReg B).surfaceBody.take 7 := rfl

end VeriTile.Bench.Examples.FloatDTypeSoftmax.Kernels
