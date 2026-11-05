import VeriTile.Triton.DSL
namespace VeriTile.Bench.Repairs.ChunkDeltaFwd
open VeriTile.Triton
structure Parameters where
  (B H T K V BT BC BK BV NT : Nat)
  (wBatch wHead wTime wDim : Nat)
  (useSeed storeFinal : Bool)

end VeriTile.Bench.Repairs.ChunkDeltaFwd
