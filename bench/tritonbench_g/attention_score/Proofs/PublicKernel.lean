import bench.tritonbench_g.attention_score.Proofs.Support.SourceDefinitions
import bench.tritonbench_g.attention_score.Proofs.Reused.CurrentExecution

import bench.tritonbench_g.attention_score.Proofs.SourceTiles

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked
open VeriTile Triton Reused Geometry
noncomputable section

/-- The unchanged selected Python JIT body, using the existing readable
quotation in Kernels.lean. The existing comparison candidate
commutes the FP32 score-scaling multiplication to scalar-left order. -/
abbrev kernel := Whole002.Execution.kernel
abbrev realExecute := Whole002.Execution.ExactReal.execute
abbrev nativeExecute := Whole002.Execution.Native.execute

/-- Existing scalar-left comparison of the same repaired FP32 score source.
It retains the selected query-length loop, masks, TF32 dot and FP32 stores. -/
def scalarLeftKernel (c : Shape) (offset size : Int) (scale : ℝ)
    (window complement : Bool) : ComputeKernel :=
  RealSource.source c.B c.D (parameters c offset size scale window complement) Bool.true

/-- The public source really is the widened shared quotation used by both
executors; no parallel handwritten AST or execution-success premise. -/
theorem shared_source (c : Shape) (offset size : Int) (scale : ℝ) (window complement : Bool) :
    kernel c offset size scale window complement = DenseSource.kernel c offset size scale window complement :=
  Whole002.Execution.kernel_eq c offset size scale window complement

theorem kernel_tiles_bounded (c : Shape) (offset size : Int) (scale : ℝ)
    (window complement : Bool) (tile : c.B*c.D≤1048576) :
    ∀ shape ∈ statementListShapes (kernel c offset size scale window complement).surfaceBody,
      shape.prod≤1048576 := by
  unfold kernel Whole002.Execution.kernel
  apply quotation_shapes_bounded
  · rcases c.block with h|h|h <;> omega
  · exact tile

/-- Invocation metadata uses the standard typed FP32 scalar argument. -/
theorem scale_argument (value : ℝ) :
    TypedSource.scaleArgument value = ScalarArgument.fp32 "sm_scale" value := rfl
#axiomsClean scale_argument

#axiomsClean shared_source
#axiomsClean kernel_tiles_bounded
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked
