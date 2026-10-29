import VeriTile.Triton.Core.Ast

namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.Parametric
open VeriTile.Triton

/-- All eleven original writer parameters. The two proofs select the audited
rank-one order-zero notation domain without removing either source binder. -/
structure Parameters where
  input : RegionName
  output : RegionName
  inputStride : Nat
  inputOrder : Nat
  outputStride : Nat
  outputOrder : Nat
  size : Nat
  numTasks : Nat
  iterations : Nat
  block : Nat
  oneTile : Bool
  inputOrderZero : inputOrder = 0
  outputOrderZero : outputOrder = 0

end VeriTile.Bench.TritonBenchG.IsfiniteKernel.Parametric
