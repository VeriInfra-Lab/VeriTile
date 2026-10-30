import bench.examples.SoftmaxReciprocal.Kernels
import bench.examples.SoftmaxReciprocal.Proofs.LegacySurface

namespace VeriTile.Bench.Examples.SoftmaxReciprocal.Kernels
open VeriTile Triton

theorem stableSoftmaxKernel_preserved (xReg yReg : RegionName) (B : Nat) :
    stableSoftmaxKernel xReg yReg B = LegacySurface.stableSoftmaxKernel xReg yReg B := rfl

theorem softmaxRecipKernel_preserved (xReg yReg : RegionName) (B : Nat) :
    softmaxRecipKernel xReg yReg B = LegacySurface.softmaxRecipKernel xReg yReg B := rfl

/-- Proof-only decomposition; connected to the actual complete source below. -/
def commonPrefix (xReg : RegionName) (B : Nat) : ComputeKernel :=
triton {
  pid  := tl.program_id(0)
  offs := pid * $(B) + tl.arange(0, $(B))
  x    := tl.load($(xReg) + offs, dtype=tl.float32)
  m    := tl.max(x, axis=0)
  e    := tl.exp(x - m)
  s    := tl.sum(e, axis=0)
}

theorem commonPrefix_source (xReg yReg : RegionName) (B : Nat) :
    (commonPrefix xReg B).surfaceBody =
      (stableSoftmaxKernel xReg yReg B).surfaceBody.take 6 := rfl

end VeriTile.Bench.Examples.SoftmaxReciprocal.Kernels
