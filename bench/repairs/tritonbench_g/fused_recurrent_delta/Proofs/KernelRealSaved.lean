import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelRealSource
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelReal
open VeriTile.Triton FP.Structural KernelRegisters KernelBackwardRegisters
open scoped BigOperators
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 16000000

def backwardColumn (s : Shape) (f : Flags) (bh iv : Nat) (u : State ℝ) (j : Fin s.BBV) : Math.Input s.BK where
  q := fun t i => if i.val<s.K then (u.mem "q" (keyAddress s bh t i.val)).read .real else 0
  k := fun t i => if i.val<s.K then (u.mem "k" (keyAddress s bh t i.val)).read .real else 0
  x := fun t => if iv*s.BBV+j.val<s.V then (u.mem "v" (valAddress s bh iv s.BBV t j.val)).read .real else 0
  beta := fun t => if f.vectorBeta then
    if iv*s.BBV+j.val<s.V then (u.mem "beta" (valAddress s bh iv s.BBV t j.val)).read .real else 0
    else (u.mem "beta" (betaAddress s bh t)).read .real
  h0 := fun i => if f.initial && decide (i.val<s.K) && decide (iv*s.BBV+j.val<s.V) then
    (u.mem "h0" (stateAddress s bh iv s.BBV i.val j.val)).read .real else 0

theorem backward_column_padding (s : Shape) (f : Flags) (bh iv : Nat) (u : State ℝ) (j : Fin s.BBV)
    (active : iv*s.BBV+j.val<s.V) :
    backwardColumn s f bh iv u j=padInput s.BK (unpaddedColumn s f bh (iv*s.BBV+j.val) u) := by
  unfold backwardColumn padInput unpaddedColumn
  congr 1
  · simp [active,valAddress,Nat.add_assoc]
  · simp [active,valAddress,Nat.add_assoc]
  · funext i
    cases hf : f.initial <;> by_cases hi : i.val<s.K <;>
      simp [padVec,hf,hi,active,stateAddress,Nat.add_assoc]

theorem backward_column_padding_state (s : Shape) (f : Flags) (bh iv : Nat) (u : State ℝ) (j : Fin s.BBV)
    (padding : ¬iv*s.BBV+j.val<s.V) (n : Nat) (i : Fin s.BK) :
    Math.state (backwardColumn s f bh iv u j) n i=0 := by
  induction n generalizing i with
  | zero => simp [Math.state,backwardColumn,padding]
  | succ n ih =>
    have old : Math.state (backwardColumn s f bh iv u j) n=fun _ => 0 := funext ih
    rw [Math.state,old]
    simp [Math.next,Math.residual,Math.dot,backwardColumn,padding]

/-- Each program starts with the same pristine memory and its own launch IDs. -/
def forwardEntry (bh iv : Nat) (pristine : State ℝ) : State ℝ :=
  {pristine with pids:=fun ax => if ax=0 then iv else if ax=1 then 0
    else if ax=2 then bh else pristine.pids ax}

/-- The family required by the saved-residual connection is constructed from
complete source execution, including when the value grid has multiple CTAs. -/
theorem forward_entries_execute (s : Shape) (f : Flags) (scale : ℝ) (bh : Nat) (pristine : State ℝ) :
    ∃ outputs : Nat → State ℝ,∀ iv,iv<s.FNV →
      KernelForward.Facts algebra s f scale bh iv (forwardEntry bh iv pristine) (outputs iv) := by
  have runs : ∀ iv,∃ v,KernelForward.Facts algebra s f scale bh iv (forwardEntry bh iv pristine) v := by
    intro iv
    exact KernelForward.complete algebra s f scale bh iv (forwardEntry bh iv pristine)
      (by simp [forwardEntry]) (by simp [forwardEntry]) (by simp [forwardEntry])
  choose outputs facts using runs
  exact ⟨outputs,fun iv _ => facts iv⟩

/-- Per-program forward results supply saved residual cells even though the
backward value block is wider. The image equation identifies actual saved
entry cells; no mathematical residual or backward output is an assumption. -/
theorem saved_from_forward (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (pristine entry : State ℝ) (outputs : Nat → State ℝ)
    (runs : ∀ iv,iv<s.FNV → KernelForward.Facts algebra s f scale bh iv (forwardEntry bh iv pristine) (outputs iv))
    (same : ∀ r : RegionName,r="q" ∨ r="k" ∨ r="beta" ∨ r="h0" → ∀ off,entry.mem r off=pristine.mem r off)
    (image : ∀ (t : Fin s.T) (j : Fin s.BBV),iv*s.BBV+j.val<s.V →
      entry.mem "v" (valAddress s bh iv s.BBV t.val j.val)=
        (outputs ((iv*s.BBV+j.val)/s.FBV)).mem "v" (valAddress s bh iv s.BBV t.val j.val)) :
    Saved s (KernelBackwardReference.input algebra s f bh iv entry) (backwardColumn s f bh iv pristine)
      (KernelBackwardReference.input algebra s f bh iv entry).incoming
      (fun j i => (KernelBackwardReference.input algebra s f bh iv entry).terminal (i,j,PUnit.unit)) := by
  refine ⟨?_,?_,?_,?_,fun _ _ => rfl,?_,fun _ _ => rfl⟩
  · intro t i j
    simp [KernelBackwardReference.input,KernelReference.read,KernelReference.cast,KernelReference.zero,algebra,
      backwardColumn,same "q" (Or.inl rfl)]
  · intro t i j
    simp [KernelBackwardReference.input,KernelReference.read,KernelReference.cast,KernelReference.zero,algebra,
      backwardColumn,same "k" (Or.inr (Or.inl rfl))]
  · intro t j
    simp [KernelBackwardReference.input,KernelReference.read,KernelReference.cast,KernelReference.zero,algebra,
      backwardColumn,same "beta" (Or.inr (Or.inr (Or.inl rfl)))]
  · intro t ht j
    by_cases active : iv*s.BBV+j.val<s.V
    · obtain ⟨inside,lane,coordinate⟩ := Geometry.owner_exists s.V s.FBV (iv*s.BBV+j.val) (Geometry.tile_pos s).2.1 active
      let jf : Fin s.FBV := ⟨(iv*s.BBV+j.val)%s.FBV,lane⟩
      have fa : ((iv*s.BBV+j.val)/s.FBV)*s.FBV+jf.val<s.V := by simpa only [jf,coordinate] using active
      have address : valAddress s bh ((iv*s.BBV+j.val)/s.FBV) s.FBV t jf.val=
          valAddress s bh iv s.BBV t j.val := by
        simp only [valAddress,←Nat.add_assoc]
        have coord : ((iv*s.BBV+j.val)/s.FBV)*s.FBV+jf.val=iv*s.BBV+j.val := coordinate
        omega
      have columns : column s f bh ((iv*s.BBV+j.val)/s.FBV)
          (forwardEntry bh ((iv*s.BBV+j.val)/s.FBV) pristine) jf=
          backwardColumn s f bh iv pristine j := by
        change column s f bh ((iv*s.BBV+j.val)/s.FBV) pristine jf=backwardColumn s f bh iv pristine j
        rw [column_padding s f bh _ pristine jf fa,backward_column_padding s f bh iv pristine j active]
        rw [show ((iv*s.BBV+j.val)/s.FBV)*s.FBV+jf.val=iv*s.BBV+j.val from coordinate]
      have result := congrArg (fun cell : Cell ℝ => cell.read .real)
        ((runs _ inside).residuals ⟨t,ht⟩ jf fa)
      dsimp only at result
      rw [stored_real,forward_residual,columns,address] at result
      simp only [KernelBackwardReference.input,KernelReference.read,active,decide_true,ite_true]
      rw [image ⟨t,ht⟩ j active]
      exact result
    · have zero : Math.state (backwardColumn s f bh iv pristine j) t=fun _ => 0 :=
        funext (backward_column_padding_state s f bh iv pristine j active t)
      rw [zero]
      simp [KernelBackwardReference.input,KernelReference.read,KernelReference.cast,KernelReference.zero,
        algebra,backwardColumn,active,Math.residual,Math.dot]
  · intro i j
    simp [KernelBackwardReference.input,KernelReference.read,KernelReference.cast,KernelReference.zero,KernelReference.add,
      algebra,backwardColumn,same "h0" (Or.inr (Or.inr (Or.inr rfl)))]
    all_goals split <;> simp_all

#axiomsClean backward_column_padding
#axiomsClean backward_column_padding_state
#axiomsClean forward_entries_execute
#axiomsClean saved_from_forward
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelReal
