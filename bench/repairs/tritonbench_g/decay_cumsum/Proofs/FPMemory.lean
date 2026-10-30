import bench.repairs.tritonbench_g.decay_cumsum.Proofs.FPPrefix

namespace VeriTile.Bench.Repairs.DecayCumsum.FPMemory
open VeriTile.Triton FP.Structural FPOps FPPrefix Geometry
noncomputable section
set_option maxHeartbeats 3000000
variable {α : Type} [Inhabited α]

theorem tile_width (h : Call) (p : Phase) (prog : Program h p) : h.BK≤h.width := by
  have hf : 0<h.features := Nat.zero_lt_of_lt prog.featureInside
  exact Nat.le_mul_of_pos_left _ hf

theorem off_succ (h : Call) (p : Phase) (prog : Program h p) (t : Nat) (j : Fin h.BK) :
    off h p prog t j+h.width=off h p prog (t+1) j := by
  simp [off,Nat.add_mul]
  omega

theorem off_unique (h : Call) (p : Phase) (prog : Program h p)
    (t t' : Nat) (j j' : Fin h.BK) (he : off h p prog t j=off h p prog t' j') :
    t=t' ∧ j=j' := by
  have width := tile_width h p prog
  have hj := j.isLt
  have hj' := j'.isLt
  have times : t=t' := by
    rcases lt_trichotomy t t' with lt|eq|lt
    · have order := Nat.mul_le_mul_right h.width (Nat.succ_le_of_lt lt)
      dsimp [off] at he
      nlinarith
    · exact eq
    · have order := Nat.mul_le_mul_right h.width (Nat.succ_le_of_lt lt)
      dsimp [off] at he
      nlinarith
  subst t'
  exact ⟨rfl,Fin.ext (by dsimp [off] at he; omega)⟩

theorem off_pred (h : Call) (p : Phase) (prog : Program h p) (t : Nat) (positive : 0<t)
    (j : Fin h.BK) : off h p prog t j-h.width=off h p prog (t-1) j := by
  have ht : t=(t-1)+1 := by omega
  rw [ht,←off_succ]
  simp

def footprint (h : Call) (p : Phase) (prog : Program h p) (regions : List RegionName)
    (r : RegionName) (o : Nat) : Prop := r∈regions ∧ ∃ t : Fin p.span, ∃ j : Fin h.BK, o=off h p prog t.val j

def Frame (h : Call) (p : Phase) (prog : Program h p) (regions : List RegionName)
    (u v : State α) : Prop := ∀ r o, ¬footprint h p prog regions r o → v.mem r o=u.mem r o

theorem cell (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p)
    (t : Nat) (r : RegionName) (value : Values α .real [h.BK]) (u : State α) (j : Fin h.BK) :
    (scatter M h.BK r (off h p prog t) value u).mem r (off h p prog t j)=
      FPReference.stored M (value (j,PUnit.unit)) := by
  apply State.scatter_readback (shape := [h.BK]) r (fun i => off h p prog t i.1)
    (fun i => FPReference.stored M (value i)) u (j,PUnit.unit)
  rintro ⟨a,⟨⟩⟩ ⟨b,⟨⟩⟩ he
  have same := (off_unique h p prog t t a b he).2
  subst b
  rfl

theorem other_step (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p)
    (t t' : Nat) (different : t≠t') (r : RegionName) (value : Values α .real [h.BK])
    (u : State α) (j : Fin h.BK) :
    (scatter M h.BK r (off h p prog t) value u).mem r (off h p prog t' j)=u.mem r (off h p prog t' j) := by
  apply State.scatter_frame
  right
  rintro ⟨k,⟨⟩⟩ _ he
  exact different (off_unique h p prog t t' k j he.symm).1

theorem frame (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p)
    (t : Fin p.span) (r : RegionName) (regions : List RegionName) (hr : r∈regions)
    (value : Values α .real [h.BK]) (u : State α) :
    Frame h p prog regions u (scatter M h.BK r (off h p prog t.val) value u) := by
  intro r' o outside
  apply State.scatter_frame
  by_cases same : r'=r
  · right
    rintro ⟨j,⟨⟩⟩ _ he
    exact outside ⟨same ▸ hr,t,j,he⟩
  · exact Or.inl same

#axiomsClean cell
#axiomsClean frame
end
end VeriTile.Bench.Repairs.DecayCumsum.FPMemory
