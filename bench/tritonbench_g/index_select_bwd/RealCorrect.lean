import bench.tritonbench_g.index_select_bwd.Proofs.LogicalCoordinates

namespace VeriTile.Bench.Repairs.IndexSelectBwd.Correct
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Two masked FP32 input channels and the in-place destination output.
The shared signature displays the actual source, channels, windows and mask. -/
def indexIO (rest : Algebra ℝ) (p : Launch) : LogicalProgramIO ℝ ℝ (Interface.SliceValues ℝ p) (Fin p.block → ℝ) :=
  Interface.logicalIO rest (· + ·) p

/-- One slice adds an incoming contribution to the current destination.
The current value may include earlier contributions from duplicate rows. -/
def mathematical (p : Launch) (values : Interface.SliceValues ℝ p) : Fin p.block → ℝ :=
  fun i => values.current i + values.incoming i

/-- Complete source correctness, actual active-access safety and exterior
frame. Real is the mathematical interpretation of the FP32 source. -/
specification index_select_bwd_kernel_correctness (rest : Algebra ℝ) (p : Launch)
    (stride : p.stride = 1) (count : p.count ≤ p.block) :
    Spec.Real (indexIO rest p ⊨ fun values =>
      fun i => values.current i + values.incoming i) :=
  Interface.logical_implements rest (· + ·) p stride count

#specNonCircular mathematical avoiding [kernel, Source.execution, FP.Structural.exec]
#axiomsClean index_select_bwd_kernel_correctness
#stmtSurfaceSubset index_select_bwd_kernel_correctness ⊆
  [Spec.Real, indexIO, Interface.SliceValues.current, Interface.SliceValues.incoming,
    LogicalProgramIO.Implements, FP.Structural.Algebra, Launch, Launch.stride, Launch.count, Launch.block, Interface.SliceValues]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IndexSelectBwd.Correct
