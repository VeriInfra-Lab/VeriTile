import bench.tritonbench_g.fused_recurrent_hgrn.Proofs.WholeGrid

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn
open VeriTile.Triton FP.Structural
namespace Execution
noncomputable section
set_option maxHeartbeats 12000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def ForwardRegion (BH T D : Nat) (S : Bool) (r : RegionName) (a : Nat) : Prop :=
  (r="o" ∧ a < BH*T*D) ∨ (S=Bool.true ∧ r="ht" ∧ a < BH*D)
def BackwardRegion (BH T D : Nat) (r : RegionName) (a : Nat) : Prop :=
  (r="dx" ∨ r="dg") ∧ a < BH*T*D

theorem forward_grid (p : Ops α) (input : Input α) (BH T D BD : Nat) (hBD : 0 < BD)
    (U S : Bool) (s : State α) (h : Inputs s input T D) :
    ∃ q, gridRun p (forward T D BD U S).surfaceBody (programs BH D BD) s=some q ∧
      Inputs q input T D ∧ Saved p q input U BH T D ∧
      (S=Bool.true → ∀ bh, bh < BH → ∀ d, d < D →
        q.mem "ht" (stateAddr bh D d)=.mk .real (state p U input bh d T)) ∧
      (∀ r a, ¬ ForwardRegion BH T D S r a → q.mem r a=s.mem r a) := by
  obtain ⟨q,he,hq,hv,hf⟩ := grid_run_complete p (forward T D BD U S).surfaceBody
    (programs BH D BD) (fun s => Inputs s input T D) (fWrites T D BD S)
    (forwardTarget p input U T D) (fun id _ s h => forward_program p input T D BD U S id s h) s h
  refine ⟨q,he,hq,?_,?_,?_⟩
  · intro bh hb t ht d hd
    obtain ⟨hblock,hlane,hsum⟩ := grid_owner D BD d hBD hd
    have res := hv "o" (rowAddr bh T D t d)
      ⟨(d/BD,bh),(programs_mem BH D BD _).mpr ⟨hblock,hb⟩,
        Or.inl ⟨rfl,t,ht,⟨d%BD,hlane⟩,by simpa [hsum] using hd,by simp [hsum]⟩⟩
    have dec := row_decode bh T D t d ht hd
    simpa [forwardTarget,dec.1,dec.2.1,dec.2.2] using res
  · intro hS bh hb d hd
    obtain ⟨hblock,hlane,hsum⟩ := grid_owner D BD d hBD hd
    have res := hv "ht" (stateAddr bh D d)
      ⟨(d/BD,bh),(programs_mem BH D BD _).mpr ⟨hblock,hb⟩,
        Or.inr ⟨hS,rfl,⟨d%BD,hlane⟩,by simpa [hsum] using hd,by simp [hsum]⟩⟩
    have dec := slot_decode bh D d hd
    simpa [forwardTarget,dec.1,dec.2] using res
  · intro r a miss
    apply hf r a
    intro id hid hw
    have hb := ((programs_mem BH D BD id).mp hid).2
    apply miss
    rcases hw with ⟨hr,t,ht,j,hj,ha⟩|⟨hS,hr,j,hj,ha⟩
    · exact Or.inl ⟨hr,ha ▸ row_bound BH T D id.2 t _ hb ht hj⟩
    · exact Or.inr ⟨hS,hr,ha ▸ state_bound BH D id.2 _ hb hj⟩

def Gradients (p : Ops α) (s : State α) (input : Input α) (U V : Bool) (BH T D : Nat) : Prop :=
  ∀ bh, bh < BH → ∀ t, t < T → ∀ d, d < D →
    s.mem "dx" (rowAddr bh T D t d)=.mk .real (dx p V input T bh d t) ∧
    s.mem "dg" (rowAddr bh T D t d)=.mk .real (dg p U V input T bh d t)

theorem backward_grid (p : Ops α) (input : Input α) (BH T D BD : Nat) (hBD : 0 < BD)
    (U V : Bool) (s : State α) (h : Inputs s input T D) (hs : Saved p s input U BH T D) :
    ∃ q, gridRun p (backward T D BD U V).surfaceBody (programs BH D BD) s=some q ∧
      Inputs q input T D ∧ Saved p q input U BH T D ∧ Gradients p q input U V BH T D ∧
      (∀ r a, ¬ BackwardRegion BH T D r a → q.mem r a=s.mem r a) := by
  obtain ⟨q,he,hq,hv,hf⟩ := grid_run_complete p (backward T D BD U V).surfaceBody
    (programs BH D BD) (fun s => Inputs s input T D ∧ Saved p s input U BH T D)
    (bWrites T D BD) (backwardTarget p input U V T D)
    (fun id hid s h => backward_program p input BH T D BD U V id
      ((programs_mem BH D BD id).mp hid).2 s h.1 h.2) s ⟨h,hs⟩
  refine ⟨q,he,hq.1,hq.2,?_,?_⟩
  · intro bh hb t ht d hd
    obtain ⟨hblock,hlane,hsum⟩ := grid_owner D BD d hBD hd
    have H : ∀ r : RegionName, r="dx" ∨ r="dg" → q.mem r (rowAddr bh T D t d)=
        backwardTarget p input U V T D r (rowAddr bh T D t d) := by
      intro r hr
      exact hv r _ ⟨(d/BD,bh),(programs_mem BH D BD _).mpr ⟨hblock,hb⟩,
        hr,t,ht,⟨d%BD,hlane⟩,by simpa [hsum] using hd,by simp [hsum]⟩
    have dec := row_decode bh T D t d ht hd
    constructor
    · simpa [backwardTarget,dec.1,dec.2.1,dec.2.2] using H "dx" (Or.inl rfl)
    · simpa [backwardTarget,dec.1,dec.2.1,dec.2.2] using H "dg" (Or.inr rfl)
  · intro r a miss
    apply hf r a
    intro id hid hw
    have hb := ((programs_mem BH D BD id).mp hid).2
    obtain ⟨hr,t,ht,j,hj,ha⟩ := hw
    exact miss ⟨hr,ha ▸ row_bound BH T D id.2 t _ hb ht hj⟩

/-- Both complete source grids. Forward and backward may select different tiles.
No execution success or output identity appears among the assumptions. -/
theorem whole_result (p : Ops α) (input : Input α) (BH T D BDF BDB : Nat)
    (hF : 0 < BDF) (hB : 0 < BDB) (U S V : Bool) (s : State α) (h : Inputs s input T D) :
    ∃ f b,
      gridRun p (forward T D BDF U S).surfaceBody (programs BH D BDF) s=some f ∧
      gridRun p (backward T D BDB U V).surfaceBody (programs BH D BDB) f=some b ∧
      Saved p b input U BH T D ∧ Gradients p b input U V BH T D ∧
      (S=Bool.true → ∀ bh, bh < BH → ∀ d, d < D →
        b.mem "ht" (stateAddr bh D d)=.mk .real (state p U input bh d T)) ∧
      (∀ r a, ¬ ForwardRegion BH T D S r a → ¬ BackwardRegion BH T D r a → b.mem r a=s.mem r a) := by
  obtain ⟨f,hf,hI,hO,hH,hff⟩ := forward_grid p input BH T D BDF hF U S s h
  obtain ⟨b,hb,hI,hO,hG,hbf⟩ := backward_grid p input BH T D BDB hB U V f hI hO
  refine ⟨f,b,hf,hb,hO,hG,?_,?_⟩
  · intro hS bh hbh d hd
    rw [hbf "ht" _ (by simp [BackwardRegion])]
    exact hH hS bh hbh d hd
  · intro r a hnf hnb
    rw [hbf r a hnb,hff r a hnf]

#axiomsClean forward_grid
#axiomsClean backward_grid
#axiomsClean whole_result
end
end Execution
end VeriTile.Bench.Repairs.FusedRecurrentHgrn
