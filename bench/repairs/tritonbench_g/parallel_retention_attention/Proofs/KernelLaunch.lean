import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.KernelBinding

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (BackwardConfig)
noncomputable section

/-- The selected backward shell's seven index assignments, expanded at the
helper-call boundary. Only launch PIDs enter; no tile result is supplied. -/
def decodedBackwardConfig (s : Shape) (pids : Nat → Nat) : BackwardConfig where
  toForwardConfig := config s 64
  do_ := "g"
  dz := "unused"
  dk := "dk"
  dv := "dv"
  i_bh := pids 2
  i_c := pids 1
  i_k := pids 0/blocks s.DV s.BV
  i_v := pids 0%blocks s.DV s.BV
  i_h := pids 2%s.H

theorem decoded_backward (s : Shape) (b : Bases) (m : Memory α) (tile : Tile)
    (valid : ProgramDomain s 64 tile) :
    decodedBackwardConfig s (initial s b m tile).pids=backwardConfig s tile := by
  have decoded := grid_decode s 64 tile valid
  simp only [decodedBackwardConfig,initial,decoded.2.1,decoded.2.2,backwardConfig]

#axiomsClean decoded_backward
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
