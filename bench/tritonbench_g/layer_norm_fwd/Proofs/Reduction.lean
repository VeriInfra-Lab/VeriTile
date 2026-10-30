import bench.tritonbench_g.layer_norm_fwd.Proofs.Domain
import VeriTile.Triton.Semantics.MaskedReduction

/-! The source's two complete padded reductions equal the independent logical
N-column mean and variance, at arbitrary block size N ≤ block. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.Reduction
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 2000000

/-- All values after the source input, scaling, dropout and residual stages.
The padding theorem applies equally to both random streams and all flags. -/
def combinedTile (f : Flags) (a : Inputs) (M N block : Nat) (p : ℝ)
    (row : Nat) : Tile .real [block] :=
  ⟨fun j => some (Formula.combined f (Formula.paddedInputs a N) M p row j.1.val)⟩

theorem combined_sum (f : Flags) (a : Inputs) (M N block : Nat)
    (p : ℝ) (row : Nat) (hN : N ≤ block) :
    (Tile.reduceSum (shape := [block]) ⟨0, by simp⟩ Bool.false
      (combinedTile f a M N block p row)).data PUnit.unit =
      some (∑ col ∈ Finset.range N, Formula.combined f a M p row col) := by
  simp only [Tile.reduceSum_false, Tile.reduceSumDrop_data, combinedTile]
  simp only [TileShape.axisDim, TileShape.insertAxisIndex]
  rw [WithBot.sum_someTerm_eq_some]
  congr 1
  change (∑ col : Fin block, Formula.combined f (Formula.paddedInputs a N) M p row col.val) = _
  rw [Fin.sum_univ_eq_sum_range (fun col => Formula.combined f (Formula.paddedInputs a N) M p row col)]
  exact Formula.sum_combined_padded f a M N block p row hN

def centeredTile (f : Flags) (a : Inputs) (M N block : Nat) (p : ℝ)
    (row : Nat) : Tile .real [block] :=
  ⟨fun j => some (if j.1.val < N then Formula.centered f a M N p row j.1.val else 0)⟩

theorem centered_square_sum (f : Flags) (a : Inputs) (M N block : Nat)
    (p : ℝ) (row : Nat) (hN : N ≤ block) :
    (Tile.reduceSum (shape := [block]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (.consSame .nil)
        (centeredTile f a M N block p row) (centeredTile f a M N block p row))).data PUnit.unit =
      some (∑ col ∈ Finset.range N, (Formula.centered f a M N p row col)^2) := by
  simp only [Tile.reduceSum_false, Tile.reduceSumDrop_data, centeredTile]
  simp only [TileShape.axisDim, TileShape.insertAxisIndex, Tile.bop, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.mul, WithBot.realMul, Option.map₂_some_some]
  rw [WithBot.sum_someTerm_eq_some]
  congr 1
  change (∑ col : Fin block,
    (if col.val < N then Formula.centered f a M N p row col.val else 0) *
    (if col.val < N then Formula.centered f a M N p row col.val else 0)) = _
  rw [Fin.sum_univ_eq_sum_range (fun col =>
    (if col < N then Formula.centered f a M N p row col else 0) *
    (if col < N then Formula.centered f a M N p row col else 0))]
  have hpoint (col : Nat) :
      (if col < N then Formula.centered f a M N p row col else 0) *
        (if col < N then Formula.centered f a M N p row col else 0) =
      if col < N then (Formula.centered f a M N p row col)^2 else 0 := by
    by_cases h : col < N <;> simp [h, pow_two]
  simp_rw [hpoint]
  exact Formula.sum_padded _ hN

theorem mean_op (f : Flags) (a : Inputs) (M N block : Nat)
    (p : ℝ) (row : Nat) (hN : N ≤ block) (s : BlockState)
    (hx : s.regs .real [block] "x" = some (combinedTile f a M N block p row)) :
    evalOp (.div .real .nil
      (.reduceSum ⟨0, by simp⟩ Bool.false (.ref .real [block] "x")) (.const (N : ℝ))) s =
      some (Tile.scalar (some (Formula.mean f a M N p row))) := by
  simp only [evalOp.eq_def, hx, bind, Option.bind, pure]
  congr 1
  ext j
  cases j
  simp only [Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex, Tile.scalar, NumericDType.div, WithBot.realDiv,
    combined_sum f a M N block p row hN, Option.map₂_some_some, Formula.mean]

theorem variance_op (f : Flags) (a : Inputs) (M N block : Nat)
    (p : ℝ) (row : Nat) (hN : N ≤ block) (s : BlockState)
    (hx : s.regs .real [block] "xbar" = some (centeredTile f a M N block p row)) :
    evalOp (.div .real .nil
      (.reduceSum ⟨0, by simp⟩ Bool.false
        (.mul .real (.consSame .nil) (.ref .real [block] "xbar") (.ref .real [block] "xbar")))
      (.const (N : ℝ))) s =
      some (Tile.scalar (some (Formula.variance f a M N p row))) := by
  simp only [evalOp.eq_def, hx, bind, Option.bind, pure]
  congr 1
  ext j
  cases j
  have hsum := centered_square_sum f a M N block p row hN
  simp only [Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex] at hsum
  simp only [Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex, Tile.scalar, NumericDType.div,
    WithBot.realDiv, hsum, Option.map₂_some_some, Formula.variance]

theorem centered_op (f : Flags) (a : Inputs) (M N block : Nat)
    (p : ℝ) (row : Nat) (s : BlockState)
    (hx : s.regs .real [block] "x" = some (combinedTile f a M N block p row))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hmean : f.rms = Bool.false → s.regs .real [] "mean" =
      some (Tile.scalar (some (Formula.mean f a M N p row)))) :
    evalOp (.where (.lt .nat .scalarR (.ref .nat [block] "cols") (.constNat N))
      (if f.rms then .ref .real [block] "x" else
        .sub .real .scalarR (.ref .real [block] "x") (.ref .real [] "mean"))
      (.broadcast (.const 0.0) [block])) s =
      some (centeredTile f a M N block p row) := by
  cases hr : f.rms
  · have hm := hmean hr
    simp only [hr, Bool.false_eq_true, if_false, evalOp.eq_def, hx, hm, hcols,
      bind, Option.bind, pure]
    congr 1
    ext j
    by_cases hj : j.1.val < N
    all_goals simp [Tile.select, Tile.bop, Tile.cop, ComparableDType.lt,
      NumericDType.sub, combinedTile, centeredTile, Formula.centered,
      Formula.combined_padded, hr, hj] <;> norm_num
  · simp only [hr, if_true, evalOp.eq_def, hx, hcols, bind, Option.bind, pure]
    congr 1
    ext j
    by_cases hj : j.1.val < N
    all_goals simp [Tile.select, Tile.cop, ComparableDType.lt,
      combinedTile, centeredTile, Formula.centered, Formula.combined_padded, hr, hj] <;> norm_num

theorem rstd_op (f : Flags) (a : Inputs) (M N : Nat)
    (eps p : ℝ) (row : Nat) (s : BlockState)
    (hvar : s.regs .real [] "var" =
      some (Tile.scalar (some (Formula.variance f a M N p row)))) :
    evalOp (.div .real .nil (.const 1)
      (.sqrt (.add .real .nil (.ref .real [] "var") (.const eps)))) s =
      some (Tile.scalar (some (Formula.rstd f a M N eps p row))) := by
  simp [evalOp.eq_def, hvar, Tile.bop, Tile.uop, NumericDType.add, NumericDType.div,
    Formula.rstd]

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.Reduction
