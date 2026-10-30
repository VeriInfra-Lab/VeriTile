import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.iv_dependent_matmul.IO
import bench.tritonbench_g.iv_dependent_matmul.Proofs.IOResults

namespace VeriTile.Bench.Repairs.IvDependentMatmulDotShape.LogicalReader
open VeriTile.Triton FP.Structural
open IvDependentMatmul
open scoped ProgramIO
noncomputable section

/-- Independent dense matrix product, including the empty K sum. -/
def formula (g : Kernel.Geometry) (vals : ChannelValues ℝ (IO.inputs g)) :
    ChannelValues ℝ (IO.outputs g) :=
  IO.result g (fun i j => ∑ k : Fin g.inner,
    IO.matrixA g vals i k.val * IO.matrixB g vals k.val j)


abbrev kernelIO (rest : Algebra ℝ) (g : Kernel.Geometry) :=
  IO.matmulIO (RealExecution.backend rest) .source g

/-- Complete selected source grid: logical matrix result, actual masked access
safety and exact exterior frame. Physical geometry and launch limits retain
the original scope; the Real initial memory has no extra dtype-tag premise. -/
theorem kernel_real (rest : Algebra ℝ) (g : Kernel.Geometry)
    (layout : IO.Layout g) (launchable : Index64.Launchable g.request) :
    Spec.Real (kernelIO rest g ⊨ formula g) :=
  IOResults.real rest g layout launchable

end
end VeriTile.Bench.Repairs.IvDependentMatmulDotShape.LogicalReader

namespace VeriTile.Bench.Repairs.IvDependentMatmulDotShape.LogicalReader
open VeriTile.Triton FP.Structural
open IvDependentMatmul
open scoped LogicalProgramIO
noncomputable section

theorem full_judgment_iff (rest : Algebra ℝ) (g : Kernel.Geometry) :
    (IO.logicalIO (RealExecution.backend rest) .source g ⊨ fun xs row col =>
      ∑ k : Fin g.inner, xs.a row k * xs.b k col) ↔ (kernelIO rest g).Implements (formula g) := by
  have target : (IO.logicalIO (RealExecution.backend rest) .source g).lift
      (fun xs row col => ∑ k : Fin g.inner, xs.a row k * xs.b k col) = formula g := by
    funext xs o; fin_cases o; funext j
    change (∑ k : Fin g.inner, ((IO.logicalInputs ℝ g) xs).a j.1 k * ((IO.logicalInputs ℝ g) xs).b k j.2.1) =
      ∑ k : Fin g.inner, IO.matrixA g xs j.1.val k.val * IO.matrixB g xs k.val j.2.1.val
    apply Finset.sum_congr rfl
    intro k _
    simp only [IO.matrixA,IO.matrixB,dif_pos j.1.isLt,dif_pos j.2.1.isLt,dif_pos k.isLt]
    rfl
  change (kernelIO rest g).Implements _ ↔ _
  simpa only [target]

end
end VeriTile.Bench.Repairs.IvDependentMatmulDotShape.LogicalReader
