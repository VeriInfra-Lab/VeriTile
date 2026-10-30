import bench.repairs.tritonbench_g.fused_rotary_embedding_tile_checked.Proofs.Layout
import VeriTile.Triton.Float.ProgramIO

namespace VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.Source
open VeriTile.Triton FP.Structural FusedRotaryEmbedding
noncomputable section
variable {α : Type} [Inhabited α]

def invocation (rest : Algebra α) (ops : Ops α) (c : Config) (pid : Program) : Invocation α where
  kernel := kernel (parameters c.shape)
  backend := algebra rest ops
  clearRegisters := Bool.true
  pids := fun ax => if ax=0 then pid.1 else if ax=1 then pid.2 else 0
  numPids := some (fun _=>1)

end
end VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.Source
