import bench.tritonbench_g.matmul_leakyrelu_fp8.Proofs.LogicalReader

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.Correct
open VeriTile.Triton FP.Structural
open MatmulLeakyreluFp8 MatmulLeakyreluFp8.Checked
open scoped BigOperators LogicalProgramIO
noncomputable section

/-- Independent matrix mathematics on logical finite input/output coordinates.
The IO retains the actual complete source grid and every original condition. -/
specification kernel_correct (M : Algebra ℝ) (law : Execution.ScalarLaws M realOps)
    (g : KernelBuffers.Geometry) (b : KernelBuffers.Buffers) (layout : KernelBuffers.Layout g b) :
    VeriTile.Spec.Real (ReaderIO.view M g b ⊨ fun xs r c =>
      let value := ∑ t : Fin g.K, xs.A r t * xs.B t c
      if g.enabled then if 0 ≤ value then value else (1 / 100 : ℝ) * value else value) :=
  ReaderProof.correct M law g b layout

#axiomsClean kernel_correct
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.Correct
