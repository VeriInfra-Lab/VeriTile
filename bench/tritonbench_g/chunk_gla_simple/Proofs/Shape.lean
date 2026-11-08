import bench.tritonbench_g.chunk_gla_simple.Proofs.Support.SourceTypes
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile
open VeriTile.Triton
namespace Host
theorem Shape.bk_min (c : Shape) : 16 ≤ c.BK := Nat.le_max_left _ _
theorem Shape.bk_max (c : Shape) : c.BK ≤ 64 := max_le (by decide) (Nat.min_le_right _ _)
theorem Shape.bk_pos (c : Shape) : 0 < c.BK := lt_of_lt_of_le (by decide) c.bk_min
end Host
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile
