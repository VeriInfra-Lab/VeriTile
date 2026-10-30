import bench.tritonbench_g.fused_recurrent_hgrn.IO
import bench.tritonbench_g.fused_recurrent_hgrn.Proofs.IOImplements
import bench.tritonbench_g.fused_recurrent_hgrn.Proofs.Gradient

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn.IO
open VeriTile.Triton FP.Structural
noncomputable section

def tensorFormula (BH T D : Nat) (U V : Bool) (xs : TensorInputs ℝ BH T D) : TensorOutputs ℝ BH T D :=
  let input := tensorData BH T D xs
  { savedState := fun bh t d => state realOps U input bh.val d.val (t.val+1)
    dx := fun bh t d => FusedRecurrentHgrn.dx realOps V input T bh.val d.val t.val
    dg := fun bh t d => FusedRecurrentHgrn.dg realOps U V input T bh.val d.val t.val
    finalState := fun bh d => state realOps U input bh.val d.val T }

def rawFormula (BH T D : Nat) (U V : Bool) (xs : ChannelValues ℝ (inputs BH T D)) :=
  let input := asInput realOps BH T D xs
  result BH T D (state realOps U input)
    (fun bh d t => FusedRecurrentHgrn.dx realOps V input T bh d t)
    (fun bh d t => FusedRecurrentHgrn.dg realOps U V input T bh d t)

theorem formula_lift (BH T D BDF BDB : Nat) (U S V : Bool) :
    (logicalIO BH T D BDF BDB U S V).lift (tensorFormula BH T D U V) = rawFormula BH T D U V := by
  funext xs o j
  fin_cases o <;>
    dsimp [LogicalProgramIO.lift, logicalIO, outputCoordinates, tensorFormula, tensorData,
      rawFormula, result]
  all_goals rw [(inputCoordinates ℝ BH T D).symm_apply_apply]
  all_goals rfl

/-- The arbitrary universally quantified property includes the reader's exact
HasDerivAt claim. No derivative or tagged/optional input is discarded. -/
theorem full_iff (BH T D BDF BDB : Nat) (U S V : Bool)
    (P : ChannelValues ℝ (inputs BH T D) → Prop) :
    ((logicalIO BH T D BDF BDB U S V).Implements (tensorFormula BH T D U V) ∧
      ∀ xs, P ((inputCoordinates ℝ BH T D).symm xs)) ↔
    ((recurrentIO realOps id BH T D BDF BDB U S V).Implements (rawFormula BH T D U V) ∧ ∀ xs, P xs) := by
  have ioEq : (logicalIO BH T D BDF BDB U S V).Implements (tensorFormula BH T D U V) ↔
      (recurrentIO realOps id BH T D BDF BDB U S V).Implements (rawFormula BH T D U V) := by
    change (recurrentIO realOps id BH T D BDF BDB U S V).Implements _ ↔ _
    rw [formula_lift]
  constructor
  · rintro ⟨h,p⟩
    refine ⟨ioEq.mp h,fun xs => ?_⟩
    simpa only [Equiv.symm_apply_apply] using p ((inputCoordinates ℝ BH T D) xs)
  · rintro ⟨h,p⟩
    exact ⟨ioEq.mpr h,fun xs => p _⟩

def tensorObjective (BH T D : Nat) (U V : Bool) (xs : TensorInputs ℝ BH T D)
    (bh d : Nat) (vx vg : Nat → ℝ) (e : ℝ) : ℝ :=
  let input := tensorData BH T D xs
  (∑ t∈Finset.range T, input "do" bh t d*pathState input U bh d vx vg e (t+1))+
    terminal realOps V input bh d*pathState input U bh d vx vg e T

theorem logical_correct (BH T D BDF BDB : Nat) (hF : 0<BDF) (hB : 0<BDB) (U S V : Bool) :
    (logicalIO BH T D BDF BDB U S V).Implements (tensorFormula BH T D U V) ∧
      ∀ xs (bh : Fin BH) (d : Fin D) (vx vg : Nat → ℝ),
        HasDerivAt (tensorObjective BH T D U V xs bh.val d.val vx vg)
          (∑ t∈Finset.range T,
            (FusedRecurrentHgrn.dx realOps V (tensorData BH T D xs) T bh.val d.val t*vx t+
             FusedRecurrentHgrn.dg realOps U V (tensorData BH T D xs) T bh.val d.val t*vg t)) 0 := by
  apply (full_iff BH T D BDF BDB U S V (fun vals =>
    ∀ (bh : Fin BH) (d : Fin D) (vx vg : Nat → ℝ),
      HasDerivAt (fun e =>
        let input := asInput realOps BH T D vals
        (∑ t∈Finset.range T, input "do" bh.val t d.val*pathState input U bh.val d.val vx vg e (t+1))+
          terminal realOps V input bh.val d.val*pathState input U bh.val d.val vx vg e T)
        (∑ t∈Finset.range T,
          (FusedRecurrentHgrn.dx realOps V (asInput realOps BH T D vals) T bh.val d.val t*vx t+
           FusedRecurrentHgrn.dg realOps U V (asInput realOps BH T D vals) T bh.val d.val t*vg t)) 0)).mpr
  refine ⟨IOImplements.implements realOps id BH T D BDF BDB hF hB U S V,?_⟩
  intro vals bh d vx vg
  exact loss_derivative (asInput realOps BH T D vals) U V T bh.val d.val vx vg
end
end VeriTile.Bench.Repairs.FusedRecurrentHgrn.IO
