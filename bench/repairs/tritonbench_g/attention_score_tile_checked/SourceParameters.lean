import VeriTile.Triton.DSL

/-! Existing caller data and complete kernel argument binding, without proof imports. -/
namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealSource
open VeriTile.Triton
structure Parameters where
  (Q K M Out : RegionName)
  (queryBatch queryHead queryRow queryDim : Nat)
  (keyBatch keyHead keyRow keyDim : Nat)
  (outputBatch outputHead outputInner : Nat)
  (batches heads kvHeads queryContext roundedContext keyContext : Nat)
  (offset size : Int)
  scale : ℝ
  (window complement evenM evenN : Bool)

end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealSource

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.Geometry
open VeriTile.Triton
noncomputable section
def Supported (B : Nat) : Prop := B = 16 ∨ B = 32 ∨ B = 64

structure Shape where
  (Z H HK N R NK D B : Nat)
  kvPositive : 0 < HK
  headDivides : H % HK = 0
  normalizerCapacity : N ≤ R
  feature : ∃ k : Nat, 4 ≤ k ∧ D = 2 ^ k
  block : Supported B

def parameters (c : Shape) (offset size : Int) (scale : ℝ)
    (window complement : Bool) : RealSource.Parameters where
  Q := "q"
  K := "k"
  M := "m"
  Out := "out"
  queryBatch := c.H * c.N * c.D
  queryHead := c.N * c.D
  queryRow := c.D
  queryDim := 1
  keyBatch := c.HK * c.NK * c.D
  keyHead := c.NK * c.D
  keyRow := c.D
  keyDim := 1
  outputBatch := c.H * c.NK
  outputHead := c.NK
  outputInner := 1
  batches := c.Z
  heads := c.H
  kvHeads := c.HK
  queryContext := c.N
  roundedContext := c.R
  keyContext := c.NK
  offset := offset
  size := size
  scale := scale
  window := window
  complement := complement
  evenM := decide (c.N % c.B = 0)
  evenN := decide (c.NK % c.B = 0)

end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.Geometry
