import bench.tritonbench_g.parallel_retention_attention.Proofs.QueryReal
import bench.tritonbench_g.parallel_retention_attention.Proofs.ReverseCausal

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.RealTarget
noncomputable section
open scoped BigOperators

/-- Independent mathematical contributions of one feature shard. The source
wrapper sums these contributions along its scratch shard dimension. -/
def forward (s : Shape) (x : Inputs ℝ) (tile : Tile) (row col : Nat) : ℝ :=
  ∑ n∈Finset.range (tile.chunk*128+row+1),
    gamma (tile.bh%s.H)^(tile.chunk*128+row-n)*forwardProduct s x tile row col n
def query (s : Shape) (x : Inputs ℝ) (tile : Tile) (row col : Nat) : ℝ :=
  (Real.sqrt (s.DK:ℝ))⁻¹*∑ n∈Finset.range (tile.chunk*64+row+1),
    gamma (tile.bh%s.H)^(tile.chunk*64+row-n)*queryProduct s x tile row col n
def key (s : Shape) (x : Inputs ℝ) (tile : Tile) (row col : Nat) : ℝ :=
  (Real.sqrt (s.DK:ℝ))⁻¹*∑ n∈Finset.range s.T,if tile.chunk*64+row≤n then
    gamma (tile.bh%s.H)^(n-(tile.chunk*64+row))*keyProduct s x tile row col n else 0
def value (s : Shape) (x : Inputs ℝ) (tile : Tile) (row col : Nat) : ℝ :=
  (Real.sqrt (s.DK:ℝ))⁻¹*∑ n∈Finset.range s.T,if tile.chunk*64+row≤n then
    gamma (tile.bh%s.H)^(n-(tile.chunk*64+row))*valueProduct s x tile row col n else 0

end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.RealTarget
