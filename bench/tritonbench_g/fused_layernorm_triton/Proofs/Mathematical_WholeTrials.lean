import bench.tritonbench_g.fused_layernorm_triton.Proofs.Mathematical_WholeGrid

namespace VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
open VeriTile.Triton FP.Structural FPStages
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

structure Configuration where
  block : Nat
  tree : ReductionTree block
  allowed : block = 1024 ∨ block = 2048

theorem Configuration.positive (c : Configuration) : 0 < c.block := by
  rcases c.allowed with h | h <;> omega

def gridMiss (Y Mean Rstd : RegionName) (S D : Nat) (q : RegionName) (off : Nat) : Prop :=
  ∀ row, row < S → rowMiss Y Mean Rstd D row q off

end
end VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
