import VeriTile.Triton.Float.LogicalProgramIO
import bench.repairs.tritonbench_g.kldiv_compute.IO
import bench.repairs.tritonbench_g.kldiv_compute.Proofs.PublicIO
import bench.repairs.tritonbench_g.kldiv_compute.Proofs.RealKernel

namespace VeriTile.Bench.Repairs.KldivCompute.Kernel.LogicalReader
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs (gridCount)
open scoped ProgramIO
noncomputable section

/-- Whole-grid FP32 source with full logical input/output vectors. Coincident
input/output regions remain allowed; each output uses the initial logical inputs.
The Structural cell channel is `.real`; the source AST retains FP32 precision. -/
def kldivIO (p : Parameters) : ProgramIO ℝ ℝ := PublicIO.io Math.algebra p

/-- Independent KL formula for every logical element, successful full-grid
source execution, backed active accesses, and the exact exterior typed-cell frame. -/
theorem correctness (p : Parameters) (positive : 0 < p.block) :
    Spec.Real (kldivIO p ⊨ fun (xs : ChannelValues ℝ (PublicIO.inputChannels p)) o => match o with
      | ⟨0,_⟩ => fun j =>
          let x : ℝ := xs ⟨0,by simp [PublicIO.inputChannels]⟩ j
          let y : ℝ := xs ⟨1,by simp [PublicIO.inputChannels]⟩ j
          x * Real.log (x/y)) := by
  have target : PublicIO.formula Math.formula p =
      (fun (xs : ChannelValues ℝ (PublicIO.inputChannels p)) o => match o with
      | ⟨0,_⟩ => fun j =>
          let x : ℝ := xs ⟨0,by simp [PublicIO.inputChannels]⟩ j
          let y : ℝ := xs ⟨1,by simp [PublicIO.inputChannels]⟩ j
          x * Real.log (x/y)) := by
    funext xs port
    rcases port with ⟨port,hp⟩
    have hz : port=0 := by have small : port<1 := hp; omega
    subst port
    rfl
  have result := PublicIO.implements Math.algebra p positive Math.formula Math.scalar_formula
  rw [target] at result
  exact result

end
end VeriTile.Bench.Repairs.KldivCompute.Kernel.LogicalReader

namespace VeriTile.Bench.Repairs.KldivCompute.Kernel.LogicalReader
open VeriTile.Triton FP.Structural
open scoped LogicalProgramIO
noncomputable section

theorem full_judgment_iff (p : Parameters) :
    (PublicIO.logicalIO Math.algebra p ⊨ fun xs i => xs.x i * Real.log (xs.x i / xs.y i)) ↔
      (PublicIO.io Math.algebra p).Implements (PublicIO.formula Math.formula p) := by
  have target : (PublicIO.logicalIO Math.algebra p).lift (fun xs i => xs.x i * Real.log (xs.x i / xs.y i)) =
      PublicIO.formula Math.formula p := by
    funext xs o; fin_cases o; funext j; rfl
  change (PublicIO.io Math.algebra p).Implements _ ↔ _
  rw [target]

end
end VeriTile.Bench.Repairs.KldivCompute.Kernel.LogicalReader
