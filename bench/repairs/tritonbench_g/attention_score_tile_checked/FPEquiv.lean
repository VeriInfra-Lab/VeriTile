import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.SourcePair

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.FPEquiv
open VeriTile Triton FP.ProfiledRewrite Reused Geometry SourcePair
open scoped VeriTile.Spec
noncomputable section

/-- The actual repaired complete FP32 source and its existing scalar-left
candidate. Only qk*qk_scale becomes qk_scale*qk at the retained finite operand
sites; signed windows, query-length loop, TF32 dot and FP32 storage remain.
The full unrestricted native grid theorem remains in Misc/NativeCorrect.lean. -/
specification score_scaling_equiv (c : Shape) (offset size : Int) (scale : ℝ)
    (window complement : Bool) (R : FP.ScalarArithmetic.Rules) :
    original c offset size scale window complement ≡[R]
      scalarLeft c offset size scale window complement :=
  SourcePair.equivalent c offset size scale window complement R

#print_fp_assumptions score_scaling_equiv
#axiomsClean score_scaling_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.FPEquiv
