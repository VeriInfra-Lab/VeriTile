import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.Grid

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn
open VeriTile.Triton FP.Structural
namespace Execution
noncomputable section
set_option maxHeartbeats 12000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

@[simp] theorem lane_enter (s : State α) (id : Program) (BD : Nat) (j : TileIndex [BD]) :
    lane (enter s id) BD j=id.1*BD+j.1.val := rfl
@[simp] theorem row_enter (s : State α) (id : Program) (t T D BD : Nat) (j : TileIndex [BD]) :
    row (enter s id) t T D BD j=rowAddr id.2 T D t (id.1*BD+j.1.val) := rfl
@[simp] theorem slot_enter (s : State α) (id : Program) (D BD : Nat) (j : TileIndex [BD]) :
    slot (enter s id) D BD j=stateAddr id.2 D (id.1*BD+j.1.val) := rfl
@[simp] theorem read_enter (s : State α) (id : Program) (r : RegionName) (a : Nat) :
    read (enter s id) r a=read s r a := rfl
@[simp] theorem enter_bh (s : State α) (id : Program) : (enter s id).pids 1=id.2 := rfl

theorem forward_program (p : Ops α) (input : Input α) (T D BD : Nat) (U S : Bool)
    (id : Program) (s : State α) (h : Inputs s input T D) :
    ∃ q, run (algebra p) (forward T D BD U S).surfaceBody (enter s id)=some q ∧
      Inputs q input T D ∧
      (∀ r a, fWrites T D BD S id r a → q.mem r a=forwardTarget p input U T D r a) ∧
      (∀ r a, ¬ fWrites T D BD S id r a → q.mem r a=s.mem r a) := by
  obtain ⟨q,he,hpid,ho,hht,hf⟩ := forward_complete p (enter s id) T D BD U S
  have frame : ∀ r a, ¬ fWrites T D BD S id r a → q.mem r a=s.mem r a := by
    intro r a miss
    exact hf r a (by simpa only [fWrites_enter] using miss)
  have vals : ∀ j : TileIndex [BD], id.1*BD+j.1.val < D → ∀ n,
      values p (enter s id) U T D BD n j=state p U input id.2 (id.1*BD+j.1.val) n := by
    intro j hj n
    apply values_eq_state p (enter s id) U T D BD input j hj
    · intro t
      simpa using h.rows "x" (by simp) id.2 t (id.1*BD+j.1.val)
    · intro t
      simpa using h.rows "g" (by simp) id.2 t (id.1*BD+j.1.val)
    · simpa using h.slots "h0" (by simp) id.2 (id.1*BD+j.1.val)
  refine ⟨q,he,?_,?_,frame⟩
  · apply inputs_frame s q input T D h
    intro r a hr
    apply frame r a
    simp only [fWrites]
    aesop
  · intro r a hw
    rcases hw with ⟨rfl,t,ht,j,hj,rfl⟩|⟨hS,rfl,j,hj,rfl⟩
    · have hv := ho t ht (j,PUnit.unit) hj
      have dec := row_decode id.2 T D t (id.1*BD+j.val) ht hj
      simpa [forwardTarget,vals (j,PUnit.unit) hj,dec.1,dec.2.1,dec.2.2] using hv
    · have hv := hht hS (j,PUnit.unit) hj
      have dec := slot_decode id.2 D (id.1*BD+j.val) hj
      simpa [forwardTarget,vals (j,PUnit.unit) hj,dec.1,dec.2] using hv

def Saved (p : Ops α) (s : State α) (input : Input α) (U : Bool) (BH T D : Nat) : Prop :=
  ∀ bh, bh < BH → ∀ t, t < T → ∀ d, d < D →
    s.mem "o" (rowAddr bh T D t d)=.mk .real (state p U input bh d (t+1))

theorem backward_program (p : Ops α) (input : Input α) (BH T D BD : Nat) (U V : Bool)
    (id : Program) (hid : id.2 < BH) (s : State α) (h : Inputs s input T D)
    (hs : Saved p s input U BH T D) :
    ∃ q, run (algebra p) (backward T D BD U V).surfaceBody (enter s id)=some q ∧
      (Inputs q input T D ∧ Saved p q input U BH T D) ∧
      (∀ r a, bWrites T D BD id r a → q.mem r a=backwardTarget p input U V T D r a) ∧
      (∀ r a, ¬ bWrites T D BD id r a → q.mem r a=s.mem r a) := by
  obtain ⟨q,he,hpid,hx,hg,hf⟩ := backward_complete p (enter s id) T D BD U V
  have frame : ∀ r a, ¬ bWrites T D BD id r a → q.mem r a=s.mem r a := by
    intro r a miss
    exact hf r a (by simpa only [bWrites_enter] using miss)
  have hdo : ∀ j : TileIndex [BD], ∀ t,
      read (enter s id) "do" (row (enter s id) t T D BD j)=input "do" id.2 t (id.1*BD+j.1.val) := by
    intro j t
    simpa using h.rows "do" (by simp) id.2 t (id.1*BD+j.1.val)
  have hgate : ∀ j : TileIndex [BD], ∀ t,
      read (enter s id) "g" (row (enter s id) t T D BD j)=input "g" id.2 t (id.1*BD+j.1.val) := by
    intro j t
    simpa using h.rows "g" (by simp) id.2 t (id.1*BD+j.1.val)
  have hv : ∀ j : TileIndex [BD], read (enter s id) "dht" (slot (enter s id) D BD j)=
      input "dht" id.2 0 (id.1*BD+j.1.val) := by
    intro j
    simpa using h.slots "dht" (by simp) id.2 (id.1*BD+j.1.val)
  have hh : ∀ j : TileIndex [BD], read (enter s id) "h0" (slot (enter s id) D BD j)=
      input "h0" id.2 0 (id.1*BD+j.1.val) := by
    intro j
    simpa using h.slots "h0" (by simp) id.2 (id.1*BD+j.1.val)
  have ho : ∀ j : TileIndex [BD], id.1*BD+j.1.val < D → ∀ t, t < T →
      read (enter s id) "o" (row (enter s id) t T D BD j)=state p U input id.2 (id.1*BD+j.1.val) (t+1) := by
    intro j hj t ht
    rw [row_enter,read_enter,read,hs id.2 hid t ht _ hj]
    rfl
  refine ⟨q,he,⟨?_,?_⟩,?_,frame⟩
  · apply inputs_frame s q input T D h
    intro r a hr
    apply frame r a
    simp only [bWrites]
    aesop
  · intro bh hb t ht d hd
    rw [frame "o" _ (by simp [bWrites])]
    exact hs bh hb t ht d hd
  · intro r a hw
    rcases hw with ⟨hr,t,ht,j,hj,rfl⟩
    have dec := row_decode id.2 T D t (id.1*BD+j.val) ht hj
    rcases hr with rfl|rfl
    · have hvx := reverseDx_eq_dx p (enter s id) V T D BD t input (j,PUnit.unit) hj
        (hdo _) (hgate _) (hv _)
      have res := hx t ht (j,PUnit.unit) hj
      rw [hvx] at res
      simpa [backwardTarget,dec.1,dec.2.1,dec.2.2] using res
    · have hvg := reverseDg_eq_dg p (enter s id) U V T D BD t input (j,PUnit.unit) hj ht
        (hdo _) (hgate _) (hv _) (hh _) (ho _ hj)
      have res := hg t ht (j,PUnit.unit) hj
      rw [hvg] at res
      simpa [backwardTarget,dec.1,dec.2.1,dec.2.2] using res

#axiomsClean forward_program
#axiomsClean backward_program
end
end Execution
end VeriTile.Bench.Repairs.FusedRecurrentHgrn
