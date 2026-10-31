import bench.tritonbench_g.fused_layernorm_triton.Proofs.LegacySource
import bench.tritonbench_g.fused_layernorm_triton.Proofs.Mathematical_WholeHost

namespace VeriTile.Bench.Repairs.FusedLayernormTriton.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.FusedLayernormTriton FPStages Whole
noncomputable section

theorem normalization (X W Bias Y : RegionName) (D B start count : Nat) :
    LegacySource.normalizeCode X W Bias Y D B start count = normalizeCode X W Bias Y D B start count := by
  induction count generalizing start with
  | zero => rfl
  | succ n ih => simp only [LegacySource.normalizeCode, normalizeCode, ih]; rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedLayernormTriton.Source
