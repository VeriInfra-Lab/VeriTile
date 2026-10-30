import bench.tritonbench_g.layer_norm_welfold.IO
import bench.tritonbench_g.layer_norm_welfold.Proofs.ReadableResults

namespace VeriTile.Bench.Repairs.LayerNormWelfold.LogicalIO
open VeriTile.Triton FP.Structural Reuse Samples
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

theorem sum_range_list (D : Nat) (f : Nat → ℝ) :
    ((List.range D).map f).sum = ∑ j : Fin D, f j.val := by
  rw [← List.sum_toFinset f (List.nodup_range (n := D))]
  simpa only [List.toFinset_range] using (Finset.sum_range f)

theorem judgment_iff (rest : Algebra ℝ) (Y Mean Rstd : RegionName) (S D : Nat) (c : Configuration) :
    (io (ideal rest) Y Mean Rstd S D c.block c.tree ⊨ fun a =>
      let mean := fun row => (∑ j : Fin D, a.x row j)/(D:ℝ)
      let inverse := fun row => 1/Real.sqrt ((∑ j : Fin D, (a.x row j-mean row)^2)/(D:ℝ)+1e-05)
      ⟨mean,inverse,fun row col => (a.x row col-mean row)*inverse row*a.weight col+a.bias col⟩) ↔
    (PairIO.layerNormIO (ideal rest) id Bool.false Y Mean Rstd S D c.block c.tree).Implements
      (IO.result Y Mean Rstd S D (Readable.realFormula D)) := by
  change (PairIO.layerNormIO (ideal rest) id Bool.false Y Mean Rstd S D c.block c.tree).Implements _ ↔ _
  apply Iff.of_eq
  congr 1
  funext xs o j
  fin_cases o <;>
    dsimp [LogicalProgramIO.lift,io,inputs,outputs]
  all_goals
    simp [LogicalProgramIO.lift,io,inputs,outputs,IO.result,Readable.realFormula,
      Math.output,Math.inverse,Math.variance,Math.mean,sum_range_list,IOProof.arrayX_at,
      IO.arrayW,IO.arrayB]
  all_goals rfl

theorem implements (rest : Algebra ℝ) (Y Mean Rstd : RegionName) (S D : Nat)
    (c : Configuration) (treeComplete : c.tree.Complete)
    (sep : Separation "X" "W" "Bias" Y Mean Rstd) :
    io (ideal rest) Y Mean Rstd S D c.block c.tree ⊨ fun a =>
      let mean := fun row => (∑ j : Fin D, a.x row j)/(D:ℝ)
      let inverse := fun row => 1/Real.sqrt ((∑ j : Fin D, (a.x row j-mean row)^2)/(D:ℝ)+1e-05)
      ⟨mean,inverse,fun row col => (a.x row col-mean row)*inverse row*a.weight col+a.bias col⟩ :=
  (judgment_iff rest Y Mean Rstd S D c).mpr (Readable.real rest Y Mean Rstd S D c treeComplete sep)
end
end VeriTile.Bench.Repairs.LayerNormWelfold.LogicalIO
