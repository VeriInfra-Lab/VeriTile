import VeriTile.Triton.Core

namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.ValueProofs
open VeriTile.Triton
structure Parameters where
  (A B output : RegionName)
  (M N K BM BN BK : Nat)

end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.ValueProofs
