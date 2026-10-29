import bench.repairs.tritonbench_g.attention_score_tile_checked.Kernels
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.CurrentExecution
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.SourceTiles

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked
open VeriTile Triton Reused Geometry
noncomputable section

abbrev realExecute := Whole002.Execution.ExactReal.execute
abbrev nativeExecute := Whole002.Execution.Native.execute

/-- The public source really is the widened shared quotation used by both
executors; no parallel handwritten AST or execution-success premise. -/
theorem shared_source (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool) :
    kernel c offset size scale window complement = DenseSource.kernel c offset size scale window complement :=
  Whole002.Execution.kernel_eq c offset size scale window complement

theorem kernel_tiles_bounded (c : Shape) (offset size : Int) (scale : ℝ)
    (window complement : Bool) (tile : c.B*c.D≤1048576) :
    ∀ shape ∈ statementListShapes (kernel c offset size scale window complement).surfaceBody,
      shape.prod≤1048576 := by
  unfold kernel
  apply quotation_shapes_bounded
  · rcases c.block with h|h|h <;> omega
  · exact tile

#axiomsClean shared_source
#axiomsClean kernel_tiles_bounded
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked
