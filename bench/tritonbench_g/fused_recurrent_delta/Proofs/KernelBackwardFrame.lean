import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelBackwardResults
import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelBackwardStateMemory
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelBackwardFrame
open VeriTile.Triton FP.Structural KernelProgram KernelRegisters KernelBackwardRegisters KernelBackwardMemory
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 14000000
variable {α : Type} [Inhabited α]

def KeyWrites (s : Shape) (bh iv : Nat) (r : RegionName) (off : Nat) : Prop :=
  (r="dk" ∨ r="dq") ∧ ∃ (t : Nat) (i : Fin s.BK),t<s.T ∧ i.val<s.K ∧ off=keyAddress s (plane s bh iv) t i.val
def ValueWrites (s : Shape) (bh iv : Nat) (r : RegionName) (off : Nat) : Prop :=
  r="dv" ∧ ∃ (t : Nat) (j : Fin s.BBV),t<s.T ∧ iv*s.BBV+j.val<s.V ∧ off=valAddress s bh iv s.BBV t j.val
def BetaWrites (s : Shape) (f : Flags) (bh iv : Nat) (r : RegionName) (off : Nat) : Prop :=
  r="dbeta" ∧ if f.vectorBeta then
    ∃ (t : Nat) (j : Fin s.BBV),t<s.T ∧ iv*s.BBV+j.val<s.V ∧ off=valAddress s (plane s bh iv) iv s.BBV t j.val
  else ∃ t : Nat,t<s.T ∧ off=plane s bh iv*s.T+t
def StateWrites (s : Shape) (f : Flags) (bh iv : Nat) (r : RegionName) (off : Nat) : Prop :=
  f.initialGrad=Bool.true ∧ r="dh0" ∧ ∃ (i : Fin s.BK) (j : Fin s.BBV),
    i.val<s.K ∧ iv*s.BBV+j.val<s.V ∧ off=stateAddress s bh iv s.BBV i.val j.val
def Writes (s : Shape) (f : Flags) (bh iv : Nat) (r : RegionName) (off : Nat) : Prop :=
  KeyWrites s bh iv r off ∨ ValueWrites s bh iv r off ∨ BetaWrites s f bh iv r off ∨ StateWrites s f bh iv r off

theorem key_miss (s : Shape) (bh iv : Nat) (r : RegionName) (off : Nat)
    (outside : ¬KeyWrites s bh iv r off) (target : RegionName) (role : target="dk" ∨ target="dq")
    (t : Nat) (inside : t<s.T) :
    r≠target ∨ ∀ i : Fin s.BK,i.val<s.K → off≠keyAddress s (plane s bh iv) t i.val := by
  by_cases different : r≠target
  · exact Or.inl different
  · right
    intro i active same
    exact outside ⟨by simpa [not_ne_iff.mp different] using role,t,i,inside,active,same⟩

theorem reverse_frame (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial : State α) (n : Nat) (bound : n≤s.T) (r : RegionName) (off : Nat)
    (outside : ¬Writes s f bh iv r off) :
    (KernelReverseMemory.memory M s f scale bh iv initial n).mem r off=initial.mem r off := by
  induction n with
  | zero => rfl
  | succ n ih =>
    have inside : s.T-1-n<s.T := by omega
    have valueMiss : r≠"dv" ∨ ∀ j : Fin s.BBV,iv*s.BBV+j.val<s.V → off≠valAddress s bh iv s.BBV (s.T-1-n) j.val := by
      by_cases different : r≠"dv"
      · exact Or.inl different
      · right
        intro j active same
        exact outside (Or.inr (Or.inl ⟨not_ne_iff.mp different,_,j,inside,active,same⟩))
    unfold KernelReverseMemory.memory betaStore
    split
    · rename_i vector
      have betaMiss : r≠"dbeta" ∨ ∀ j : Fin s.BBV,iv*s.BBV+j.val<s.V →
          off≠valAddress s (plane s bh iv) iv s.BBV (s.T-1-n) j.val := by
        by_cases different : r≠"dbeta"
        · exact Or.inl different
        · right
          intro j active same
          apply outside
          right; right; left
          exact ⟨not_ne_iff.mp different,by simp only [vector,ite_true]; exact ⟨_,j,inside,active,same⟩⟩
      rw [KernelMemory.rowStore_frame M s "dbeta" _ _ _ _ _ _ r off betaMiss]
      rw [KernelMemory.rowStore_frame M s "dv" _ _ _ _ _ _ r off valueMiss,
        key_frame M s "dk" _ _ _ _ r off
          (key_miss s bh iv r off (fun h => outside (Or.inl h)) "dk" (Or.inl rfl) _ inside)]
      exact ih (by omega)
    · rename_i scalar
      have betaMiss : r≠"dbeta" ∨ off≠plane s bh iv*s.T+(s.T-1-n) := by
        by_cases different : r≠"dbeta"
        · exact Or.inl different
        · right
          intro same
          apply outside
          right; right; left
          exact ⟨not_ne_iff.mp different,by simp only [scalar,ite_false]; exact ⟨_,inside,same⟩⟩
      rw [State.write_other _ _ _ _ _ _ betaMiss]
      rw [KernelMemory.rowStore_frame M s "dv" _ _ _ _ _ _ r off valueMiss,
        key_frame M s "dk" _ _ _ _ r off
          (key_miss s bh iv r off (fun h => outside (Or.inl h)) "dk" (Or.inl rfl) _ inside)]
      exact ih (by omega)

theorem recompute_frame (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (initial : State α) (n : Nat) (bound : n≤s.T) (r : RegionName) (off : Nat)
    (outside : ¬Writes s f bh iv r off) :
    (KernelRecomputeReferenceMemory.memory M s f scale bh iv initial n).mem r off=initial.mem r off := by
  induction n with
  | zero => rfl
  | succ n ih =>
    unfold KernelRecomputeReferenceMemory.memory
    rw [key_frame M s "dq" _ _ _ _ r off
        (key_miss s bh iv r off (fun h => outside (Or.inl h)) "dq" (Or.inr rfl) n (by omega)),
      key_frame M s "dk" _ _ _ _ r off
        (key_miss s bh iv r off (fun h => outside (Or.inl h)) "dk" (Or.inl rfl) n (by omega))]
    exact ih (by omega)

#axiomsClean key_miss
#axiomsClean reverse_frame
#axiomsClean recompute_frame
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelBackwardFrame
