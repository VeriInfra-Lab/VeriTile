import VeriTile.Triton.DSL

namespace VeriTile.Bench.Repairs.ReluStridedBuffer
open Lean VeriTile.Triton

structure View where
  region : RegionName
  base : Int
  stride : Int
structure Parameters where
  input : View
  output : View
  n : Nat
  B : Nat
  tiles : Nat
  oneTile : Bool


end VeriTile.Bench.Repairs.ReluStridedBuffer
