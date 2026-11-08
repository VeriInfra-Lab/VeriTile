import bench.tritonbench_g.chunk_retention_ops.Proofs.Sources.Geometry
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001

/-- The repaired launch's chunk geometry makes every proof-clock chunk start
nonnegative and within T, including the empty-input case. The actual signed
subtraction/minimum therefore agrees with the independent natural length. -/
theorem signed_tail_length (c : Shape) (n : Nat) :
    min (64 : Int) ((c.T : Int)-((c.chunks-1-n : Nat) : Int)*64)=
      (c.length (c.chunks-1-n) : Int) := by
  have start : (c.chunks-1-n)*64≤c.T := by
    by_cases empty : c.chunks=0
    · have zero := c.chunks_zero.mp empty
      simp [empty,zero]
    · have live : c.chunks-1-n<c.chunks := by omega
      exact Nat.le_of_lt (c.chunk_start _ live)
  have remaining : (c.T : Int)-((c.chunks-1-n : Nat) : Int)*64=
      ((c.T-(c.chunks-1-n)*64 : Nat) : Int) := by omega
  rw [remaining]
  unfold Shape.length
  norm_cast

#axiomsClean signed_tail_length
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
