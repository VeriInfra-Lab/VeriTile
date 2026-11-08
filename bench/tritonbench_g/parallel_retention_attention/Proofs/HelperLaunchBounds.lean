import bench.tritonbench_g.parallel_retention_attention.Proofs.LaunchBounds
import bench.tritonbench_g.parallel_retention_attention.Proofs.BackwardBounds

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
  (queryChunk keyChunk valueChunk)

theorem helper_launch_bounds (s : Shape) (tile : Tile)
    (shape : ShapeContract s) (valid : ProgramDomain s 64 tile) :
    BackwardIndexBounds (backwardConfig s tile) := by
  let bases : Bases:=⟨0,0,0,0⟩
  let memory : Memory Nat:=fun _=>0
  let state:=initial s bases memory tile
  have h:=launch_index_bounds s tile 64 shape valid (Or.inl rfl) state rfl rfl
  have chunks:=source_chunks s 64 bases memory tile valid
  constructor
  · constructor
    · exact h.time
    · have key:=h.key
      rw [show keyChunk (config s 64) state=tile.key*s.BK from chunks.2.1] at key
      exact key
    · have value:=h.value
      rw [show valueChunk (config s 64) state=tile.value*s.BV from chunks.2.2] at value
      exact value
  · change (s.T+32-1)/32*32+32≤2147483648
    have upper:=blocks_upper s.T 32 (by decide)
    have cover:=blocks_cover s.T 128 (by decide)
    have time:=shape.indices.time_padded
    norm_num at time
    simp only [blocks] at upper cover time
    omega

#axiomsClean helper_launch_bounds
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
