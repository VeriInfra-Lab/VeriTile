import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardMath
import VeriTile.Triton.Semantics.MaskedReduction

/-! Draft for root validation: connect the independent scalar specification
to the actual extended-real tile operators. The specification in ForwardMath
does not mention tiles or the interpreter. These are proof-only bridge terms.
-/
namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward
open VeriTile.Triton
open scoped BigOperators
noncomputable section

def valuesTile {D : Nat} (values : Fin D → WithBot ℝ) : Tile .real [D] :=
  ⟨fun i => values i.1⟩

def maximumTile {D : Nat} (values : Fin D → WithBot ℝ) : Tile .real [] :=
  ⟨fun _ => rowMaximum values⟩

theorem reduceMax_valuesTile {D : Nat} (hD : 0 < D)
    (values : Fin D → WithBot ℝ) :
    Tile.reduceMax (shape := [D]) ⟨0, by simp⟩ Bool.false (valuesTile values) =
      some (maximumTile values) := by
  simp [Tile.reduceMax, Tile.reduceMaxDrop, TileShape.axisDim,
    TileShape.eraseAxis, TileShape.insertAxisIndex, hD,
    valuesTile, maximumTile, rowMaximum, Finset.sup'_eq_sup]
  rfl

def shiftedTile {D : Nat} (values : Fin D → WithBot ℝ) : Tile .real [D] :=
  Tile.bop (NumericDType.sub .real) .scalarR
    (valuesTile values) (maximumTile values)

def numeratorTile {D : Nat} (values : Fin D → WithBot ℝ) : Tile .real [D] :=
  Tile.uop WithBot.realExp (shiftedTile values)

def denominatorTile {D : Nat} (values : Fin D → WithBot ℝ) : Tile .real [] :=
  Tile.reduceSum (shape := [D]) ⟨0, by simp⟩ Bool.false (numeratorTile values)

theorem shiftedTile_data {D : Nat} (values : Fin D → WithBot ℝ) (i : Fin D) :
    (shiftedTile values).data (i, PUnit.unit) = shifted values i := by
  rfl

theorem numeratorTile_data {D : Nat} (values : Fin D → WithBot ℝ) (i : Fin D) :
    (numeratorTile values).data (i, PUnit.unit) = some (numerator values i) := by
  change WithBot.realExp (shifted values i) = some (numerator values i)
  unfold numerator
  cases shifted values i <;> rfl

theorem denominatorTile_data {D : Nat} (values : Fin D → WithBot ℝ) :
    (denominatorTile values).data PUnit.unit = some (denominator values) := by
  change @Finset.sum (Fin D) (WithBot ℝ) _ Finset.univ
      (fun i => (numeratorTile values).data (i, PUnit.unit)) =
    (some (∑ i : Fin D, numerator values i) : WithBot ℝ)
  calc
    @Finset.sum (Fin D) (WithBot ℝ) _ Finset.univ
        (fun i => (numeratorTile values).data (i, PUnit.unit)) =
      @Finset.sum (Fin D) (WithBot ℝ) _ Finset.univ
        (fun i => (some (numerator values i) : WithBot ℝ)) := by
      apply Finset.sum_congr rfl
      intro i _
      exact numeratorTile_data values i
    _ = (some (∑ i : Fin D, numerator values i) : WithBot ℝ) :=
      WithBot.sum_someTerm_eq_some Finset.univ (numerator values)

/-- The source's exact LOG and division tails after a successful maximum.
The surrounding execution proof must separately discharge D > 0. -/
def resultTile {D : Nat} (log : Bool) (values : Fin D → WithBot ℝ) : Tile .real [D] :=
  if log then
    Tile.bop (NumericDType.sub .real) .scalarR (shiftedTile values)
      (Tile.uop WithBot.realLog (denominatorTile values))
  else
    Tile.bop (NumericDType.div .real) .scalarR (numeratorTile values)
      (denominatorTile values)

theorem resultTile_extendedFormula (K row D : Nat) (log causal : Bool)
    (mode : MaskMode) (xs masks : Fin D → ℝ) (i : Fin D) :
    (resultTile log (score K row causal mode xs masks)).data (i, PUnit.unit) =
      extendedFormula K row log causal mode xs masks i := by
  cases log
  · change WithBot.realDiv
        ((numeratorTile (score K row causal mode xs masks)).data (i, PUnit.unit))
        ((denominatorTile (score K row causal mode xs masks)).data PUnit.unit) = _
    rw [numeratorTile_data, denominatorTile_data]
    rfl
  · change WithBot.realSub
        ((shiftedTile (score K row causal mode xs masks)).data (i, PUnit.unit))
        (WithBot.realLog
          ((denominatorTile (score K row causal mode xs masks)).data PUnit.unit)) = _
    rw [shiftedTile_data, denominatorTile_data]
    change WithBot.realSub (shifted (score K row causal mode xs masks) i)
        (some (Real.log (denominator (score K row causal mode xs masks)))) =
      Option.map (fun z => z - Real.log (denominator (score K row causal mode xs masks)))
        (shifted (score K row causal mode xs masks) i)
    cases shifted (score K row causal mode xs masks) i <;> rfl

end
#axiomsClean reduceMax_valuesTile
#axiomsClean shiftedTile_data
#axiomsClean numeratorTile_data
#axiomsClean denominatorTile_data
#axiomsClean resultTile_extendedFormula
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward
