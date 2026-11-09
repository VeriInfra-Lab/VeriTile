import bench.tritonbench_g.parallel_retention_attention.Proofs.KernelBinding
import bench.tritonbench_g.parallel_retention_attention.Proofs.Contract
import bench.tritonbench_g.parallel_retention_attention.Proofs.IndexBounds

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
  (queryChunk keyChunk valueChunk)

/-- The int32 facts are consequences of the existing Python shape guards,
the actual grid coordinate and the two original launch tile sizes. -/
theorem launch_index_bounds (s : Shape) (tile : Tile) (L : Nat)
    (shape : ShapeContract s) (valid : ProgramDomain s L tile)
    (tileSize : L=64 ∨ L=128) (state : State α)
    (pid0 : state.pids 0=ProgramDomain.gridX s tile)
    (pid1 : state.pids 1=tile.chunk) :
    ForwardIndexBounds (config s L) state := by
  have decoded := grid_decode s L tile valid
  have kt := Nat.mul_le_mul_right s.BK (Nat.succ_le_iff.mpr valid.key)
  have vt := Nat.mul_le_mul_right s.BV (Nat.succ_le_iff.mpr valid.value)
  have time := shape.indices.time_padded
  have key := shape.indices.key_padded
  have value := shape.indices.value_padded
  have clock : tile.chunk*L+L≤blocks s.T 128*128 := by
    have room := Nat.mul_le_mul_right L (Nat.succ_le_iff.mpr valid.chunk)
    rcases tileSize with small | large
    · subst L
      have ceiling : blocks s.T 64*64≤blocks s.T 128*128 := by
        simp only [blocks]
        omega
      nlinarith
    · subst L
      nlinarith
  constructor
  · change state.pids 1*L+L+32≤2147483648
    rw [pid1]
    norm_num at time
    omega
  · change state.pids 0/blocks s.DV s.BV*s.BK+s.BK≤2147483648
    rw [pid0,decoded.2.1]
    norm_num at key
    nlinarith
  · change state.pids 0%blocks s.DV s.BV*s.BV+s.BV≤2147483648
    rw [pid0,decoded.2.2]
    norm_num at value
    nlinarith

#axiomsClean launch_index_bounds
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
