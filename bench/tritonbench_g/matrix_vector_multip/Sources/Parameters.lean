import VeriTile.Triton.DSL.CallSignature
namespace VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed
open VeriTile.Triton
structure Parameters where
  (a b out : RegionName)
  (rows columns strideRow strideColumn strideVector strideOut blockRows blockColumns : Nat)
end VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed
