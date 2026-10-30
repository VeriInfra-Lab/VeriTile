import bench.tritonbench_g.kldiv_triton.Proofs.LogicalReader
namespace VeriTile.Bench.Repairs.KldivTriton.Kernel.Correct
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KldivTriton.FullMath (outputCount)
open scoped LogicalProgramIO
noncomputable section
/-- The raw-target branch uses the probability-zero extension. -/
def loss (flag : Bool) (prediction target : ℝ) : ℝ :=
  if flag then Real.exp target * (target-prediction)
  else if target=0 then 0 else target*(Real.log target-prediction)

/-- Independent point losses or per-chunk sums; no host post-reduction. -/
def target (N B : Nat) (flag : Bool) (reduction : Nat)
    (prediction target : Nat → ℝ) (c : Nat) : ℝ :=
  if reduction=0 then loss flag (prediction c) (target c)
  else ∑ lane : Fin B, if c*B+lane.val<N then
    loss flag (prediction (c*B+lane.val)) (target (c*B+lane.val)) else 0



/-- Full forward grid with independent point/chunk mathematics, signed strides and every complete row order. -/
specification forward (rows sy st so N B : Nat) (flag : Bool) (reduction : Nat) (positive : 0<B)
    (unique : Function.Injective (IO.rowAddress (rows:=rows) (columns:=outputCount N B reduction) so))
    (work : List Nat) (order : work.Perm (List.range rows)) :
    Spec.Real (IO.logicalForwardIO RealModel.model rows sy st so N B flag reduction work ⊨ fun xs row col =>
      target N B flag reduction (IO.extendRow xs.prediction row) (IO.extendRow xs.target row) col.val) :=
  (LogicalReader.forward_full_iff rows sy st so N B flag reduction work).mpr
    (LogicalReader.forward rows sy st so N B flag reduction positive unique work order)

/-- Full unscaled prediction derivative grid and the exact exterior frame. -/
specification backward (rows so st N B : Nat) (flag : Bool) (positive : 0<B)
    (unique : Function.Injective (IO.rowAddress (rows:=rows) (columns:=N) so))
    (work : List Nat) (order : work.Perm (List.range rows)) :
    Spec.Real (IO.logicalBackwardIO RealModel.model rows so st N B flag work ⊨ fun xs row col =>
      if flag then -Real.exp (xs row col) else -(xs row col)) :=
  (LogicalReader.backward_full_iff rows so st N B flag work).mpr
    (LogicalReader.backward rows so st N B flag positive unique work order)
#axiomsClean forward
#axiomsClean backward

end
end VeriTile.Bench.Repairs.KldivTriton.Kernel.Correct
