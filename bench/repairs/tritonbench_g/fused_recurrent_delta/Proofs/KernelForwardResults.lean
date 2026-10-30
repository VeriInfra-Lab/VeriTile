import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelForwardReferenceMemory
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelStateMemory
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelForwardResults
open VeriTile.Triton FP.Structural KernelProgram KernelRegisters KernelMemory
noncomputable section
set_option maxHeartbeats 12000000
variable {α : Type} [Inhabited α]

theorem output (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial : State α) (n t : Nat) (inside : t<n) (j : Fin s.FBV) (active : iv*s.FBV+j.val<s.V) :
    (KernelForwardReferenceMemory.memory M s f scale bh iv initial n).mem "o"
      (valAddress s bh iv s.FBV t j.val)=
        KernelReference.stored M (KernelReference.output M s (KernelReference.forwardInput M s f bh iv initial) scale t j) := by
  induction n with
  | zero => omega
  | succ n ih =>
    unfold KernelForwardReferenceMemory.memory
    by_cases current : t=n
    · subst t
      exact rowStore_cell M s "o" bh iv s.FBV n _ _ j active
    · rw [rowStore_other_time M s "o" bh iv s.FBV n t _ _ j active current,
        rowStore_frame M s "v" _ _ _ _ _ _ "o" _ (Or.inl (by decide))]
      exact ih (by omega)

theorem residual (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial : State α) (n t : Nat) (inside : t<n) (j : Fin s.FBV) (active : iv*s.FBV+j.val<s.V) :
    (KernelForwardReferenceMemory.memory M s f scale bh iv initial n).mem "v"
      (valAddress s bh iv s.FBV t j.val)=
        KernelReference.stored M (KernelReference.residual M s (KernelReference.forwardInput M s f bh iv initial) t j) := by
  induction n with
  | zero => omega
  | succ n ih =>
    unfold KernelForwardReferenceMemory.memory
    rw [rowStore_frame M s "o" _ _ _ _ _ _ "v" _ (Or.inl (by decide))]
    by_cases current : t=n
    · subst t
      exact rowStore_cell M s "v" bh iv s.FBV n _ _ j active
    · rw [rowStore_other_time M s "v" bh iv s.FBV n t _ _ j active current]
      exact ih (by omega)

def RowWrites (s : Shape) (bh iv n : Nat) (r : RegionName) (off : Nat) : Prop :=
  (r="v" ∨ r="o") ∧ ∃ (t : Nat) (j : Fin s.FBV),t<n ∧ iv*s.FBV+j.val<s.V ∧ off=valAddress s bh iv s.FBV t j.val
def Writes (s : Shape) (f : Flags) (bh iv : Nat) (r : RegionName) (off : Nat) : Prop :=
  RowWrites s bh iv s.T r off ∨
  (f.final=Bool.true ∧ r="ht" ∧ ∃ (i : Fin s.BK) (j : Fin s.FBV),
    i.val<s.K ∧ iv*s.FBV+j.val<s.V ∧ off=stateAddress s bh iv s.FBV i.val j.val)

theorem frame (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial : State α) (n : Nat) (r : RegionName) (off : Nat) (outside : ¬RowWrites s bh iv n r off) :
    (KernelForwardReferenceMemory.memory M s f scale bh iv initial n).mem r off=initial.mem r off := by
  induction n with
  | zero => rfl
  | succ n ih =>
    have miss (target : RegionName) (targetRole : target="v" ∨ target="o") :
        r≠target ∨ ∀ j : Fin s.FBV,iv*s.FBV+j.val<s.V → off≠valAddress s bh iv s.FBV n j.val := by
      by_cases different : r≠target
      · exact Or.inl different
      · right
        intro j active same
        exact outside ⟨by simpa [not_ne_iff.mp different] using targetRole,n,j,by omega,active,same⟩
    have previous : ¬RowWrites s bh iv n r off := by
      rintro ⟨role,t,j,ht,hj,eq⟩
      exact outside ⟨role,t,j,by omega,hj,eq⟩
    unfold KernelForwardReferenceMemory.memory
    rw [rowStore_frame M s "o" _ _ _ _ _ _ r off (miss "o" (Or.inr rfl)),
      rowStore_frame M s "v" _ _ _ _ _ _ r off (miss "v" (Or.inl rfl))]
    exact ih previous
#axiomsClean output
#axiomsClean residual
#axiomsClean frame
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelForwardResults
