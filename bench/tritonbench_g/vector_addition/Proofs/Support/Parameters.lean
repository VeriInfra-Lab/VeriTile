import VeriTile.Triton.Source

namespace VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs
open VeriTile.Triton

structure Parameters where
  (x y output : RegionName)
  (n block : Nat)

end VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs
