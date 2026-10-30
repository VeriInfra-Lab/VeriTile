import bench.tritonbench_g.layer_norm_triton.IO
import bench.tritonbench_g.layer_norm_triton.Proofs.ForwardRealIO
import bench.tritonbench_g.layer_norm_triton.Proofs.GroupedRealIO
import bench.tritonbench_g.layer_norm_triton.Proofs.FinalRealIO

namespace VeriTile.Bench.Repairs.LayerNormTriton.LogicalProof
open VeriTile.Triton FP.Structural SourceModel Interface Logical
noncomputable section

/-- Pure logical forward target; the reader displays its independent mathematics. -/
def forwardFormula (N : Nat) (eps : ℝ) (xs : ForwardInputs N) : ForwardOutputs N :=
  ⟨fun j => RealProof.Spec.output N xs.xAt xs.weightAt xs.biasAt eps j.val,
    RealProof.Spec.mean N xs.xAt,RealProof.Spec.inverse N xs.xAt eps⟩

theorem forward_judgment_iff (m : Model ℝ) (stride N B row : Nat) (eps : ℝ) :
    (forwardIO m stride N B row eps).Implements (forwardFormula N eps) ↔
    (Forward.io m stride N B row eps (fun _ _ => id)).Implements (ForwardRealIO.formula N eps) := by
  have same : (forwardIO m stride N B row eps).lift (forwardFormula N eps) =
      ForwardRealIO.formula N eps := by
    funext raw port
    fin_cases port <;> rfl
  change (Forward.io m stride N B row eps (fun _ _ => id)).Implements _ ↔ _
  rw [same]

theorem forward_implements (m : Model ℝ) (laws : SourceForwardReal.Laws m)
    (stride N B row : Nat) (eps : ℝ) (positive : 0<N) (covers : N≤B) :
    (forwardIO m stride N B row eps).Implements (forwardFormula N eps) :=
  (forward_judgment_iff m stride N B row eps).2
    (ForwardRealIO.implements m laws stride N B row eps positive covers)

def groupedFormula (M N G B group : Nat) (xs : GroupedInputs M N B) : GroupedOutputs M N B :=
  ⟨fun row j => RealProof.Spec.gradient N (extend (xs.xRow row.val))
      (extend (xs.dyRow row.val)) (extend xs.weight)
      (xs.mean row) (xs.inverse row) j.val,
    fun j => ((groupRows M G group).map (fun row => xs.dyRow row j *
      ((xs.xRow row j-xs.meanAt row)*xs.inverseAt row))).sum,
    fun j => ((groupRows M G group).map (fun row => xs.dyRow row j)).sum⟩

theorem xRow_eq (M N B : Nat) (raw : Grouped.Inputs ℝ M N B) (row : Nat) :
    (groupedInputs M N B raw).xRow row = Grouped.x raw row := by
  funext j
  dsimp only [GroupedInputs.xRow,Grouped.x]
  split <;> rfl

theorem dyRow_eq (M N B : Nat) (raw : Grouped.Inputs ℝ M N B) (row : Nat) :
    (groupedInputs M N B raw).dyRow row = Grouped.dy raw row := by
  funext j
  dsimp only [GroupedInputs.dyRow,Grouped.dy]
  split <;> rfl

/-- The fixed judgment keeps actual source execution, readback dtype/defaults,
visited DX result mask, complete typed outputs, writable mask and exact frame. -/
theorem grouped_judgment_iff (m : Model ℝ) (stride M N G B group : Nat) :
    (groupedIO m stride M N G B group).Implements (groupedFormula M N G B group) ↔
    (GroupedRealIO.io m stride M N G B group).Implements (GroupedRealIO.formula M N G B group) := by
  have same : (groupedIO m stride M N G B group).lift (groupedFormula M N G B group) =
      GroupedRealIO.formula M N G B group := by
    funext raw port
    fin_cases port <;> funext i
    all_goals
      dsimp only [VeriTile.Triton.LogicalIO.lift,groupedIO,groupedFormula,groupedOutputs,
        GroupedRealIO.formula,Grouped.result]
      simp only [xRow_eq,dyRow_eq,Grouped.mean,Grouped.inverse,dif_pos i.1.isLt] <;> rfl
  change (GroupedRealIO.io m stride M N G B group).Implements _ ↔ _
  rw [same]

theorem grouped_implements (m : Model ℝ) (laws : SourceForwardReal.Laws m)
    (stride M N G B group : Nat) (groups : 0<G) (covers : N≤B)
    (rowAddresses : ∀ row,row<M → row*stride=row*N) :
    (groupedIO m stride M N G B group).Implements (groupedFormula M N G B group) :=
  (grouped_judgment_iff m stride M N G B group).2
    (GroupedRealIO.implements m laws stride M N G B group groups covers rowAddresses)

def finalFormula (M BN : Nat) (xs : FinalInputs M BN) : FinalOutputs BN :=
  ⟨fun col => ∑ row : Fin M, xs.dw row col,fun col => ∑ row : Fin M, xs.db row col⟩

theorem final_judgment_iff (m : Model ℝ) (M N BN pid : Nat) :
    (finalIO m M N BN pid).Implements (finalFormula M BN) ↔
    (Final.io m M N 32 BN pid (fun _ _ => id)).Implements (FinalRealIO.formula M BN) := by
  have same : (finalIO m M N BN pid).lift (finalFormula M BN) = FinalRealIO.formula M BN := by
    funext raw port
    fin_cases port
    · funext i
      change (∑ row : Fin M, (show ℝ from raw (0 : Fin 2) (row,i.1,()))) =
        ∑ row ∈ Finset.range M, Final.value raw (0 : Fin 2) row i.1
      rw [← Fin.sum_univ_eq_sum_range]
      apply Finset.sum_congr rfl
      intro row _
      simp only [Final.value,dif_pos row.isLt]
    · funext i
      change (∑ row : Fin M, (show ℝ from raw (1 : Fin 2) (row,i.1,()))) =
        ∑ row ∈ Finset.range M, Final.value raw (1 : Fin 2) row i.1
      rw [← Fin.sum_univ_eq_sum_range]
      apply Finset.sum_congr rfl
      intro row _
      simp only [Final.value,dif_pos row.isLt]
  change (Final.io m M N 32 BN pid (fun _ _ => id)).Implements _ ↔ _
  rw [same]

theorem final_implements (m : Model ℝ) (laws : SourceFinalReal.Laws m) (M N BN pid : Nat) :
    (finalIO m M N BN pid).Implements (finalFormula M BN) :=
  (final_judgment_iff m M N BN pid).2 (FinalRealIO.implements m laws M N BN pid)

end
end VeriTile.Bench.Repairs.LayerNormTriton.LogicalProof
