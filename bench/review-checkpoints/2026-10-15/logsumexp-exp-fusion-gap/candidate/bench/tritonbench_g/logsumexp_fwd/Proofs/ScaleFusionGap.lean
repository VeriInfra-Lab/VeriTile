import bench.tritonbench_g.logsumexp_fwd.Proofs.ExpGapReachable

namespace VeriTile.Bench.TritonBenchG.LogsumexpFwd.WordGraph
open VeriTile.Triton FP.Structural FP.Equational
open SoftmaxReducev.FPValue (treeReduce maximumWord)
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def fusionInput : Word := FP.Value.ofBits .fp32 0x3f800001
/-- The host scalar is exactly representable as binary32. -/
def fusionScale : ℝ := 1 - 1 / 8388608

def fusionScalar (r : ℝ) : Word :=
  if r = fusionScale then FP.Value.ofBits .fp32 0x3f7ffffe else FP.Value.qNaN .fp32

def one : Word := FP.Value.ofBits .fp32 0x3f800000

theorem fusion_operands_decoded :
    fusionInput.toRat? = some (1 + 1 / 8388608 : Rat) ∧
    (FP.Value.ofBits .fp32 0x3f7ffffe).toRat? = some (1 - 1 / 8388608 : Rat) := by
  decide +kernel

theorem scaled_fusion_row :
    scaled .ieeeValue fusionScalar fusionScale (probeInitial fusionInput) "x" 17 32 Bool.true =
      probeRow one := by
  simp only [scaled,Bool.true_eq,if_true,loaded_probe]
  funext i
  simp only [probeRow]
  split_ifs
  all_goals simp only [fusionScalar,if_pos rfl]
  all_goals decide +kernel

theorem fusion_row_maximum : probeMaximum one = one := by
  decide +kernel

/-- Actual source separate multiply/subtract and observed fused prefix differ
on a legal full masked row. This is not a final LSE-output or GPU theorem. -/
theorem source_and_fused_prefix_differ :
    exponentArguments .ieeeValue fusionScalar ReductionPlan.withSeed fusionScale
      (probeInitial fusionInput) "x" 17 32 Bool.true (⟨1,by decide⟩,()) = FP.Value.zero .fp32 ∧
    FP.fma .ieeeValue (fusionScalar fusionScale) fusionInput (probeMaximum one).neg =
      FP.Value.ofBits .fp32 0xa8800000 ∧
    FP.Value.zero .fp32 ≠ FP.Value.ofBits .fp32 0xa8800000 := by
  simp only [exponentArguments,scaled_fusion_row]
  change FP.sub .ieeeValue (probeRow one (⟨1,by decide⟩,())) (probeMaximum one) = _ ∧ _
  rw [fusion_row_maximum]
  simp only [probeRow,if_pos (by decide : 1 < 17),if_pos rfl,fusionScalar,if_pos rfl]
  decide +kernel

#axiomsClean fusion_operands_decoded
#axiomsClean scaled_fusion_row
#axiomsClean fusion_row_maximum
#axiomsClean source_and_fused_prefix_differ
end
end VeriTile.Bench.TritonBenchG.LogsumexpFwd.WordGraph
