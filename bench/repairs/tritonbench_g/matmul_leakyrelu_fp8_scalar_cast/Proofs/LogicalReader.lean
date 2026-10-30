import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8_scalar_cast.Proofs.IOResults
import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8_scalar_cast.IO

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.ReaderProof
open VeriTile.Triton FP.Structural
open MatmulLeakyreluFp8 MatmulLeakyreluFp8.Checked
open ReaderIO
open scoped BigOperators LogicalProgramIO
noncomputable section

/-- Exact equality to the previous independent target, for every raw input. -/
theorem target_eq (M : Algebra ℝ) (g : KernelBuffers.Geometry) (b : KernelBuffers.Buffers) : (view M g b).lift (fun xs r c =>
      let value := ∑ t : Fin g.K, xs.A r t * xs.B t c
      if g.enabled then if 0 ≤ value then value else (1 / 100 : ℝ) * value else value) =
    (fun vals => IO.result g (Target.mathematical g (IO.matrixA g vals) (IO.matrixB g vals))) := by
  funext xs o j
  change ChannelValues ℝ [⟨.real,[g.M,g.K],"A"⟩,⟨.real,[g.K,g.N],"B"⟩] at xs
  fin_cases o
  rcases j with ⟨r,c,⟨⟩⟩
  change (let value := ∑ t : Fin g.K, (ReaderIO.inputs g xs).A r t * (ReaderIO.inputs g xs).B t c
    if g.enabled then if 0 ≤ value then value else (1 / 100 : ℝ) * value else value) =
      IO.result g (Target.mathematical g (IO.matrixA g xs) (IO.matrixB g xs)) (0 : Fin 1) (r,c,())
  have same : (∑ t : Fin g.K, (ReaderIO.inputs g xs).A r t * (ReaderIO.inputs g xs).B t c) =
      ∑ t ∈ Finset.range g.K, IO.matrixA g xs r.val t * IO.matrixB g xs t c.val := by
    rw [← Fin.sum_univ_eq_sum_range]
    apply Finset.sum_congr rfl
    intro t _
    simp [ReaderIO.inputs, IO.matrixA, IO.matrixB, r.isLt, c.isLt, t.isLt] <;> rfl
  simp only [same]
  rfl

/-- The complete old/new judgments are equivalent, including source execution,
source safety, every output and the exact exterior frame. -/
theorem judgment_iff (M : Algebra ℝ) (g : KernelBuffers.Geometry) (b : KernelBuffers.Buffers) :
    (view M g b ⊨ fun xs r c =>
      let value := ∑ t : Fin g.K, xs.A r t * xs.B t c
      if g.enabled then if 0 ≤ value then value else (1 / 100 : ℝ) * value else value) ↔
      (IO.matmulIO M realOps g b).Implements (fun vals => IO.result g (Target.mathematical g (IO.matrixA g vals) (IO.matrixB g vals))) := by
  rw [LogicalProgramIO.implements_iff, target_eq] <;> rfl

theorem correct (M : Algebra ℝ) (law : Execution.ScalarLaws M realOps)
    (g : KernelBuffers.Geometry) (b : KernelBuffers.Buffers) (layout : KernelBuffers.Layout g b) :
    (view M g b ⊨ fun xs r c =>
      let value := ∑ t : Fin g.K, xs.A r t * xs.B t c
      if g.enabled then if 0 ≤ value then value else (1 / 100 : ℝ) * value else value) :=
  (judgment_iff M g b).mpr (IOResults.real M law g b layout)
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.ReaderProof
