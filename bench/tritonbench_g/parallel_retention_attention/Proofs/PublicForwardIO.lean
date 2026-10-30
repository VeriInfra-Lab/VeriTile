import bench.tritonbench_g.parallel_retention_attention.Proofs.PublicSourceMemory
import bench.tritonbench_g.parallel_retention_attention.Proofs.PublicObservedValues

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.PublicIO
open VeriTile.Triton FP.Structural Kernel Kernel.Native
noncomputable section
set_option maxHeartbeats 2000000
variable {α : Type}

/-- Compose actual source access safety, source execution, observed values and
the full cell frame into the shared IO judgment. -/
theorem forward_implements (a : Scalar α) (p : DotTrees) (s : Shape) (tile : Tile)
    (valid : ProgramDomain s 128 tile) :
    (forwardIO a p s tile).Implements (forwardFormula a p s tile) := by
  intro xs _ bounds bounded state binding
  obtain ⟨final,ran,memory⟩ := forward_run s tile valid state
  refine ⟨forward_schedule_safe a p s tile valid xs bounds bounded state,final,ran,?_,?_⟩
  · intro o i active
    fin_cases o
    change mapCell (evaluate a p) (final.mem "o" (forwardAddress s tile i)) =
      FP.Structural.Cell.mk .real (forwardFormula a p s tile xs (0 : Fin 1) i)
    change activeTile s.T s.DV (tile.chunk*128) (tile.value*s.BV) i at active
    have cell : final.mem "o" (forwardAddress s tile i) = FP.Structural.Cell.mk .real
        (FPStages.forwardValues algebra dots (config s 128) ((forwardInvocation α s tile).initial state) i) := by
      rw [memory]
      exact storeTile_read "o" ((tile.bh+s.B*s.H*tile.key)*(s.T*s.DV)) s.T s.DV
        (tile.chunk*128) (tile.value*s.BV) _ _ i active.1 active.2
    rw [cell]
    change FP.Structural.Cell.mk .real (evaluate a p _) = _
    rw [forward_observed a p s tile valid xs state binding i]
  · intro r off outside
    rw [memory]
    apply storeTile_frame_full
    intro i active
    exact outside.1 (0 : Fin 1) i active

#axiomsClean forward_implements
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.PublicIO
