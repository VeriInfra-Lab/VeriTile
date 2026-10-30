import bench.tritonbench_g.fused_recurrent_hgrn.Proofs.Backward

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn
open VeriTile.Triton FP.Structural
namespace Execution
noncomputable section
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

abbrev Program := Nat × Nat
def enter (s : State α) (id : Program) : State α :=
  {s with pids := fun ax => if ax=0 then id.1 else if ax=1 then id.2 else s.pids ax}

def gridRun (p : Ops α) (body : List ComputeStmt) : List Program → State α → Option (State α)
  | [],s => some s
  | id::ids,s => (run (algebra p) body (enter s id)).bind (gridRun p body ids)

/-- A generic sequential grid theorem. Each step must prove execution, values,
input preservation, and every cell outside its actual write set. -/
theorem grid_run_complete (p : Ops α) (body : List ComputeStmt)
    (ids : List Program) (I : State α → Prop) (W : Program → RegionName → Nat → Prop)
    (target : RegionName → Nat → Cell α)
    (one : ∀ id ∈ ids, ∀ s, I s → ∃ q, run (algebra p) body (enter s id)=some q ∧ I q ∧
      (∀ r a, W id r a → q.mem r a=target r a) ∧
      (∀ r a, ¬ W id r a → q.mem r a=s.mem r a))
    (s : State α) (hI : I s) :
    ∃ q, gridRun p body ids s=some q ∧ I q ∧
      (∀ r a, (∃ id ∈ ids, W id r a) → q.mem r a=target r a) ∧
      (∀ r a, (∀ id ∈ ids, ¬ W id r a) → q.mem r a=s.mem r a) := by
  induction ids generalizing s with
  | nil => exact ⟨s,rfl,hI,by simp,by simp⟩
  | cons id ids ih =>
    obtain ⟨q,he,hq,hv,hf⟩ := one id (by simp) s hI
    obtain ⟨z,hz,hI,hvs,hfs⟩ := ih (fun j hj => one j (by simp [hj])) q hq
    refine ⟨z,by simp [gridRun,he,hz],hI,?_,?_⟩
    · intro r a h
      by_cases hs : ∃ j ∈ ids, W j r a
      · exact hvs r a hs
      · rw [hfs r a (by intro j hj hw; exact hs ⟨j,hj,hw⟩)]
        obtain ⟨j,hj,hw⟩ := h
        simp only [List.mem_cons] at hj
        rcases hj with hj|hj
        · subst j
          exact hv r a hw
        · exact False.elim (hs ⟨j,hj,hw⟩)
    · intro r a miss
      rw [hfs r a (fun j hj => miss j (by simp [hj])),hf r a (miss id (by simp))]

def programs (BH D BD : Nat) : List Program :=
  (List.range ((D+BD-1)/BD)).flatMap (fun block => (List.range BH).map (fun bh => (block,bh)))

theorem programs_mem (BH D BD : Nat) (id : Program) :
    id ∈ programs BH D BD ↔ id.1 < (D+BD-1)/BD ∧ id.2 < BH := by
  rcases id with ⟨block,bh⟩
  simp [programs]

theorem row_decode (bh T D t d : Nat) (ht : t < T) (hd : d < D) :
    rowAddr bh T D t d/D/T=bh ∧ (rowAddr bh T D t d/D)%T=t ∧ rowAddr bh T D t d%D=d := by
  have hD : 0 < D := by omega
  have hT : 0 < T := by omega
  simp [rowAddr,Nat.add_div,Nat.div_eq_of_lt hd,Nat.div_eq_of_lt ht,
    Nat.mod_eq_of_lt hd,Nat.mod_eq_of_lt ht,Nat.not_le.mpr hd,Nat.not_le.mpr ht,hD,hT]

theorem slot_decode (bh D d : Nat) (hd : d < D) : stateAddr bh D d/D=bh ∧ stateAddr bh D d%D=d := by
  have hD : 0 < D := by omega
  simp [stateAddr,Nat.add_div,Nat.div_eq_of_lt hd,Nat.mod_eq_of_lt hd,Nat.not_le.mpr hd,hD]

structure Inputs (s : State α) (input : Input α) (T D : Nat) : Prop where
  rows : ∀ r, r="x" ∨ r="g" ∨ r="do" → ∀ bh t d,
    read s r (rowAddr bh T D t d)=input r bh t d
  slots : ∀ r, r="h0" ∨ r="dht" → ∀ bh d,
    read s r (stateAddr bh D d)=input r bh 0 d

theorem inputs_frame (s q : State α) (input : Input α) (T D : Nat) (h : Inputs s input T D)
    (preserve : ∀ r a, (r="x" ∨ r="g" ∨ r="do" ∨ r="h0" ∨ r="dht") → q.mem r a=s.mem r a) :
    Inputs q input T D := by
  constructor
  · intro r hr bh t d
    rw [read,preserve r _ (by aesop)]
    exact h.rows r hr bh t d
  · intro r hr bh d
    rw [read,preserve r _ (by aesop)]
    exact h.slots r hr bh d

def forwardTarget (p : Ops α) (input : Input α) (U : Bool) (T D : Nat) (r : RegionName) (a : Nat) : Cell α :=
  .mk .real (if r="o" then state p U input (a/D/T) (a%D) ((a/D)%T+1)
    else state p U input (a/D) (a%D) T)

def backwardTarget (p : Ops α) (input : Input α) (U V : Bool) (T D : Nat) (r : RegionName) (a : Nat) : Cell α :=
  .mk .real (if r="dx" then dx p V input T (a/D/T) (a%D) ((a/D)%T)
    else dg p U V input T (a/D/T) (a%D) ((a/D)%T))

def fWrites (T D BD : Nat) (S : Bool) (id : Program) (r : RegionName) (a : Nat) : Prop :=
  (r="o" ∧ ∃ t, t < T ∧ ∃ j : Fin BD, id.1*BD+j.val < D ∧ a=rowAddr id.2 T D t (id.1*BD+j.val)) ∨
  (S=Bool.true ∧ r="ht" ∧ ∃ j : Fin BD, id.1*BD+j.val < D ∧ a=stateAddr id.2 D (id.1*BD+j.val))

def bWrites (T D BD : Nat) (id : Program) (r : RegionName) (a : Nat) : Prop :=
  (r="dx" ∨ r="dg") ∧ ∃ t, t < T ∧ ∃ j : Fin BD, id.1*BD+j.val < D ∧ a=rowAddr id.2 T D t (id.1*BD+j.val)

theorem fWrites_enter (s : State α) (id : Program) (T D BD : Nat) (S : Bool) (r : RegionName) (a : Nat) :
    ForwardWrites (enter s id) T D BD S r a ↔ fWrites T D BD S id r a := by
  simp [ForwardWrites,fWrites,enter,row,slot,lane,rowAddr,stateAddr,
    VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.channel,
    VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.rowOffset,
    VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.stateOffset]

theorem bWrites_enter (s : State α) (id : Program) (T D BD : Nat) (r : RegionName) (a : Nat) :
    BackwardWrites (enter s id) T D BD r a ↔ bWrites T D BD id r a := by
  simp [BackwardWrites,bWrites,enter,row,lane,rowAddr,
    VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.channel,
    VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.rowOffset]

end
end Execution
end VeriTile.Bench.Repairs.FusedRecurrentHgrn
