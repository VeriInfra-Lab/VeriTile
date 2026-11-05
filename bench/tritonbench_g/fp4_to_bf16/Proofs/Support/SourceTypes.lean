import VeriTile.Triton

namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue
open VeriTile.Triton
structure Parameters where
  input : RegionName
  output : RegionName
  count : Nat
  block : Nat
end VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue
