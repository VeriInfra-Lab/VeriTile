import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_Source

/-! Complete-source lowering checks for the symbolic fp32 profile. These
certify that no fallback kernel is used; execution and source-profile audits
remain distinct obligations. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Triton
set_option maxHeartbeats 5000000
set_option maxRecDepth 4000

theorem original_toAlgorithm_supported (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block) :
    ∃ algorithm, (original f r stride M N eps p block hBlock).toAlgorithm? =
      Except.ok algorithm := by
  exact ⟨_, rfl⟩

theorem candidate_toAlgorithm_supported (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block) :
    ∃ algorithm, (candidate f r stride M N eps p block hBlock).toAlgorithm? =
      Except.ok algorithm := by
  exact ⟨_, rfl⟩

end VeriTile.Bench.TritonBenchG.LayerNormFwd
