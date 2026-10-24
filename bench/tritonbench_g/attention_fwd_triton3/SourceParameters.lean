import VeriTile.Triton.DSL

/-! Source metadata shared with the retained proof modules. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
open VeriTile.Triton

namespace RealHeader
structure Strides where
  qBatch : Nat
  qHead : Nat
  kBatch : Nat
  kHead : Nat
  vBatch : Nat
  vHead : Nat
  oBatch : Nat
  oHead : Nat

end RealHeader

namespace RealPointers
structure Layout where
  Q : RegionName
  K : RegionName
  V : RegionName
  M : RegionName
  Out : RegionName
  L : RegionName
  queryContext : Nat
  roundedContext : Nat
  kvContext : Nat
  qRow : Nat
  qDim : Nat
  kDim : Nat
  kRow : Nat
  vRow : Nat
  vDim : Nat
  oRow : Nat
  oDim : Nat

end RealPointers

namespace RealPrologue
structure Parameters where
  heads : Nat
  kvHeads : Nat
  strides : RealHeader.Strides
  layout : RealPointers.Layout
  scale : ℝ
  init : Nat
  evenM : Nat
  evenN : Nat
  window : Nat
  complement : Nat
  offset : Int
  size : Int

end RealPrologue

end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
