import bench.repairs.tritonbench_g.chunk_delta_fwd.Model
namespace VeriTile.Bench.Repairs.ChunkDeltaFwd
open VeriTile Triton FP.Structural
noncomputable section
variable {α : Type}
theorem SameInputs.refl (m : Memory α) : SameInputs m m := by intro r hr o; rfl
theorem SameInputs.trans {a b c : Memory α} (ab : SameInputs a b) (bc : SameInputs b c) : SameInputs a c := by
  intro r hr o
  exact (ab r hr o).trans (bc r hr o)

theorem scatter_frame {ι : Type} (xs : List ι) (dest : RegionName) (active : ι → Prop)
    [DecidablePred active] (offset : ι → Nat) (value : ι → Cell α) (m : Memory α)
    (r : RegionName) (o : Nat)
    (safe : r≠dest ∨ ∀ i∈xs, active i → o≠offset i) :
    scatter xs dest active offset value m r o=m r o := by
  induction xs generalizing m with
  | nil => rfl
  | cons i rest ih =>
    have tail : r≠dest ∨ ∀ j∈rest, active j → o≠offset j := by
      rcases safe with h|h
      · exact Or.inl h
      · exact Or.inr (fun j hj ha => h j (by simp [hj]) ha)
    change scatter rest dest active offset value
      (if active i then fun r' o' => if r'=dest ∧ o'=offset i then value i else m r' o' else m) r o=_
    rw [ih _ tail]
    by_cases hi : active i
    · simp only [hi,if_true]
      rcases safe with hr|ho
      · simp [hr]
      · simp [ho i (by simp) hi]
    · simp [hi]

theorem scatter_inputs {ι : Type} (xs : List ι) (dest : RegionName)
    (hd : dest∉(["key","weight","value","initial"] : List RegionName)) (active : ι → Prop)
    [DecidablePred active] (offset : ι → Nat) (value : ι → Cell α) (m : Memory α) :
    SameInputs (scatter xs dest active offset value m) m := by
  intro r hr o
  apply scatter_frame
  left
  intro he
  exact hd (he ▸ hr)

theorem history_inputs (M : Algebra α) (p : Parameters) (pid : PID) (c : Nat)
    (value : HValues p α) (m : Memory α) :
    SameInputs (writeHistory M p pid c value m) m :=
  scatter_inputs _ _ (by decide) _ _ _ _

theorem corrected_inputs (M : Algebra α) (p : Parameters) (pid : PID) (c b : Nat)
    (value : CIndex p → α) (m : Memory α) :
    SameInputs (writeCorrected M p pid c b value m) m :=
  scatter_inputs _ _ (by decide) _ _ _ _

theorem final_inputs (M : Algebra α) (p : Parameters) (pid : PID)
    (value : HValues p α) (m : Memory α) :
    SameInputs (writeFinal M p pid value m) m := by
  unfold writeFinal
  split
  · exact scatter_inputs _ _ (by decide) _ _ _ _
  · exact SameInputs.refl _

#axiomsClean scatter_frame
#axiomsClean history_inputs
#axiomsClean corrected_inputs
#axiomsClean final_inputs
end
end VeriTile.Bench.Repairs.ChunkDeltaFwd
