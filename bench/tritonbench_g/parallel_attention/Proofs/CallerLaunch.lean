import bench.tritonbench_g.parallel_attention.Proofs.CallerCombined
import bench.tritonbench_g.parallel_attention.Proofs.Support.Grid

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.Caller
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section

theorem config_launch (shape : Shape) (scale : ℝ) (tile : Tile) (s : State α) :
    config (dqConfig shape scale tile) s=dqConfig shape scale (launchTile shape s):=rfl

theorem config_program (shape : Shape) (scale : ℝ) (p : Program shape 64) (s : State α) :
    config (dqConfig shape scale p.tile) (withProgram p s)=dqConfig shape scale p.tile := by
  rw [config_launch,program_tile]

#axiomsClean config_program
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.Caller
