import bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.NativeIOSafety

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.NativeInterface.Proof
open VeriTile Triton FP.Structural Reused Geometry NativeInstructions NativeStaging
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 2000000

theorem implements (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool)
    (hOffsets : c.N ≠ 0 → c.outputSize ≠ 0 → KeyOffsetsFit c) :
    (programIO c offset size scale window complement).ImplementsOnCells (initialOutput c)
      (formula c offset size scale window complement) := by
  intro xs domain bounds bounded s binding
  refine ⟨?_,SafetyProof.values c offset size scale window complement xs domain s binding hOffsets⟩
  by_cases he : RealInterface.empty c
  · simp [programIO,he,ScheduleSafe]
  · simpa only [programIO,if_neg he] using SafetyProof.schedule_safe c offset size scale window complement
      (WholeGrid.jobs c) s bounds (SafetyProof.capacities c offset size scale window complement xs bounds bounded)
      (hOffsets (fun h => he (Or.inl h)) (fun h => he (Or.inr h)))

#axiomsClean implements
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.NativeInterface.Proof
