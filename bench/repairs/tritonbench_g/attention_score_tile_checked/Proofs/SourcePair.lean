import bench.repairs.tritonbench_g.attention_score_tile_checked.PairIO

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.SourcePair
open VeriTile Triton FP.Structural FP.ProfiledRewrite Reused Geometry
open scoped VeriTile.Spec
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000

/-- The actual signed loop has its reached-operand guard; a parser mismatch
must not silently turn the source domain into an empty list. -/
theorem required_site_present (c : Shape) (offset size : Int) (scale : ℝ)
    (window complement : Bool) :
    (requirements c offset size scale window complement).length = 1 := by
  rfl

/-- Source-specific contextual proof for the already specified scalar-left
candidate. The only numeric leaf is shared strict FP32 multiplication
commutativity; no native FMA or TF32 implementation is re-proved. -/
theorem equivalent (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool)
    (R : FP.ScalarArithmetic.Rules) :
    original c offset size scale window complement ≡[R]
      scalarLeft c offset size scale window complement := by
  equiv_decompose
  all_goals fp_prove

#axiomsClean required_site_present
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.SourcePair
