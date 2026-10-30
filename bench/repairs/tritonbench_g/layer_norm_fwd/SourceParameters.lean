import VeriTile.Triton.DSL.CallSignature

namespace VeriTile.Bench.Repairs.LayerNormFwd.Source
open VeriTile.Triton

structure Flags where
  rms : Bool
  residual : Bool
  residualOut : Bool
  bias : Bool
  dropout : Bool
  dropoutMask : Bool
  rowscale : Bool
  x1 : Bool
  w1 : Bool
  b1 : Bool
  deriving DecidableEq

structure Regions where
  x : RegionName
  y : RegionName
  weight : RegionName
  bias : RegionName
  residual : RegionName
  x1 : RegionName
  weight1 : RegionName
  bias1 : RegionName
  y1 : RegionName
  residualOut : RegionName
  rowscale : RegionName
  seeds : RegionName
  dropoutMask : RegionName
  mean : RegionName
  rstd : RegionName

structure Strides where
  x : Nat
  y : Nat
  residual : Nat
  residualOut : Nat
  x1 : Nat
  y1 : Nat

abbrev sourceType := Flags → Regions → Strides → Nat → Nat → ℝ → ℝ →
  (block : Nat) → Random.ArangeSizeDomain block → ComputeKernel

end VeriTile.Bench.Repairs.LayerNormFwd.Source
