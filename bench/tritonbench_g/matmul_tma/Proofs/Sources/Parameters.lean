import VeriTile.Triton.Source

namespace VeriTile.Bench.TritonBenchG.MatmulTma.FPValue
open VeriTile.Triton

structure Parameters where
  (a b c : RegionName)
  (rows columns inner aRow aInner bInner bColumn cRow cColumn
    blockRows blockColumns blockInner : Nat)

end VeriTile.Bench.TritonBenchG.MatmulTma.FPValue
