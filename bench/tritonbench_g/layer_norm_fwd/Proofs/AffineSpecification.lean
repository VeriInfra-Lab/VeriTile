import bench.tritonbench_g.layer_norm_fwd.Proofs.AffinePrimitives
import bench.tritonbench_g.layer_norm_fwd.Proofs.InputDomain

/-! The source's affine arithmetic and coefficient loads agree with the
independent initial-snapshot formulas on every stored column. Inactive x_hat
and coefficient lanes keep their actual values and are excluded by the stores. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.Affine
open VeriTile.Triton
noncomputable section

def mathematicalXhat (f : Flags) (a : Inputs) (M N : Nat) (eps p : ℝ) (row : Nat) : Nat → ℝ :=
  normalizedValues f.rms (Formula.combined f (Formula.paddedInputs a N) M p row)
    (Formula.mean f a M N p row) (Formula.rstd f a M N eps p row)

theorem mathematicalXhat_active (f : Flags) (a : Inputs) (M N : Nat) (eps p : ℝ)
    (row col : Nat) (hcol : col < N) :
    mathematicalXhat f a M N eps p row col = Formula.normalized f a M N eps p row col := by
  cases hr : f.rms
  all_goals simp [mathematicalXhat, normalizedValues, Formula.combined_padded, hcol,
    Formula.normalized, Formula.centered, hr]

theorem affineValues_y (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (row col : Nat) (s u : BlockState)
    (hread : Domain.ReadsUnchanged f r s u) (hcol : col < N) :
    affineValues f.bias (mathematicalXhat f (snapshot r stride u) M N eps p row)
      (coefficients r.weight N s) (coefficients r.bias N s) col =
      Formula.y f (snapshot r stride u) M N eps p row col := by
  have hw := Domain.readsUnchanged_readMem f r s u hread r.weight
    (by simp [Domain.readRegion]) col
  have hb (h : f.bias = Bool.true) := Domain.readsUnchanged_readMem f r s u hread r.bias
    (by simp [Domain.readRegion, h]) col
  cases hf : f.bias
  all_goals simp [affineValues, coefficients, hcol, mathematicalXhat_active _ _ _ _ _ _ _ _ hcol,
    Formula.y, snapshot, hf, hw, hb]

theorem affineValues_y1 (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (row col : Nat) (s u : BlockState)
    (hread : Domain.ReadsUnchanged f r s u) (hcol : col < N) (hactive : f.w1 = Bool.true) :
    affineValues f.b1 (mathematicalXhat f (snapshot r stride u) M N eps p row)
      (coefficients r.weight1 N s) (coefficients r.bias1 N s) col =
      Formula.y1 f (snapshot r stride u) M N eps p row col := by
  have hw := Domain.readsUnchanged_readMem f r s u hread r.weight1
    (by simp [Domain.readRegion, hactive]) col
  have hb (h : f.b1 = Bool.true) := Domain.readsUnchanged_readMem f r s u hread r.bias1
    (by simp [Domain.readRegion, hactive, h]) col
  cases hf : f.b1
  all_goals simp [affineValues, coefficients, hcol, mathematicalXhat_active _ _ _ _ _ _ _ _ hcol,
    Formula.y1, snapshot, hf, hw, hb]

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.Affine
