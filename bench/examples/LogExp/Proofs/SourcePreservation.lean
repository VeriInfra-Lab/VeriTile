import bench.examples.LogExp.Kernels
import bench.examples.LogExp.Proofs.LegacySurface

namespace VeriTile.Bench.Examples.LogExp
open VeriTile Triton

theorem originalKernel_preserved (xReg yReg : RegionName) (B : Nat) :
    originalKernel xReg yReg B = LegacySurface.originalKernel xReg yReg B := rfl

theorem optimizedKernel_preserved (xReg yReg : RegionName) (B : Nat) :
    optimizedKernel xReg yReg B = LegacySurface.optimizedKernel xReg yReg B := rfl

end VeriTile.Bench.Examples.LogExp
