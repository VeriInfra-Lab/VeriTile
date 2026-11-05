import VeriTile.Triton

namespace VeriTile.Bench.Repairs.PowScalarTensor.Checked
open VeriTile.Triton
structure View where
  region : RegionName
  base : Int
  stride : Int
structure Parameters where
  input : View
  output : View
  n : Nat
  base : ℝ
  B : Nat
  tiles : Nat
  oneTile : Bool

end VeriTile.Bench.Repairs.PowScalarTensor.Checked
