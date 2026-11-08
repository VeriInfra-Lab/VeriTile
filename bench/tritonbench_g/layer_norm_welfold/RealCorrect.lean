import bench.tritonbench_g.layer_norm_welfold.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.LayerNormWelfold.Correct
open VeriTile.Triton FP.Structural Reuse Samples
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Complete streaming row grid with independent population statistics and
normalization over every logical feature. FP32 inputs/statistics and BF16 Y,
configuration, complete tree, separation and exact exterior frame are retained. -/
specification kernel_real (rest : Algebra ℝ) (S D : Nat) (Y Mean Rstd : RegionName)
    (c : Configuration) (treeComplete : c.tree.Complete)
    (sep : Separation "X" "W" "Bias" Y Mean Rstd) :
    VeriTile.Spec.Real (LogicalIO.io (ideal rest) Y Mean Rstd S D c.block c.tree ⊨ fun a =>
      let mean := fun row => (∑ j : Fin D, a.x row j)/(D:ℝ)
      let inverse := fun row => 1/Real.sqrt ((∑ j : Fin D, (a.x row j-mean row)^2)/(D:ℝ)+1e-05)
      ⟨mean,inverse,fun row col => (a.x row col-mean row)*inverse row*a.weight col+a.bias col⟩) :=
  LogicalIO.implements rest Y Mean Rstd S D c treeComplete sep

#axiomsClean kernel_real
end
end VeriTile.Bench.Repairs.LayerNormWelfold.Correct
