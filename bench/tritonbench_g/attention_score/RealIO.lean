import bench.tritonbench_g.attention_score.IO

/-! Full score-grid Real IO with the original input roles and empty dispatch.
The already allocated zero output is an explicit input only when the source
grid is skipped. No new initializer, wrapper executor or source is introduced. -/
namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.RealInterface
open VeriTile Triton Reused Geometry Physical
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 1000000

/-- The mathematical sum has N terms and uses the supplied normalizer M. -/
def formula (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool)
    (xs : Inputs ℝ c) : Outputs ℝ c := fun _ i =>
  let coord := outputCoordinates c i
  RealPhysical.logicalTarget c (logicalInputs c xs) offset size scale window complement coord.1 coord.2.1 coord.2.2

#specNonCircular formula avoiding [kernel,realExecute,Triton.exec]
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.RealInterface
