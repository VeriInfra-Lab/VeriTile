import bench.tritonbench_g.int_scaled_matmul.Sources.GroupedGrid
namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.GridExecution
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000

def atProgram {α : Type} (s : State α) (pid : Nat) : State α :=
  {s with pids := fun ax => if ax=0 then pid else 0}
def sequence {α : Type} (one : Nat → State α → Option (State α)) : List Nat → State α → Option (State α)
  | [],s => some s
  | pid::rest,s => (one pid s).bind (sequence one rest)
def Outside {α : Type} (output : RegionName) (initial current : State α) : Prop :=
  ∀ r o, r≠output → current.mem r o=initial.mem r o

def Post {α I : Type} (output : RegionName) (addr : I → Nat) (value : I → Cell α)
    (owns : I → Prop) (initial final : State α) : Prop :=
  (∀ x, owns x → final.mem output (addr x)=value x) ∧
  (∀ r o, (r≠output ∨ ∀ x, owns x → o≠addr x) → final.mem r o=initial.mem r o)

theorem compose {α I : Type} (output : RegionName) (addr : I → Nat)
    (hinj : Function.Injective addr) (value : I → Cell α) (owns : Nat → I → Prop)
    (one : Nat → State α → Option (State α)) (initial : State α) (count : Nat)
    (single : ∀ pid, pid<count → ∀ current, Outside output initial current →
      ∃ final, one pid current=some final ∧ Post output addr value (owns pid) current final)
    (programs : List Nat) (hprograms : ∀ pid∈programs, pid<count)
    (current : State α) (hout : Outside output initial current) :
    ∃ final, sequence one programs current=some final ∧
      Post output addr value (fun x => ∃ pid∈programs, owns pid x) current final := by
  classical
  induction programs generalizing current with
  | nil =>
    refine ⟨current,rfl,?_,?_⟩
    · simp
    · intro r o h; rfl
  | cons pid rest ih =>
    obtain ⟨mid,hm,hmout,hmframe⟩ := single pid (hprograms pid (by simp)) current hout
    have hmid : Outside output initial mid := by
      intro r o h
      exact (hmframe r o (Or.inl h)).trans (hout r o h)
    obtain ⟨final,hf,hfout,hfframe⟩ := ih (fun j hj => hprograms j (by simp [hj])) mid hmid
    refine ⟨final,?_,?_,?_⟩
    · simp only [sequence,hm,Option.bind_some,hf]
    · intro x hx
      by_cases ht : ∃ j∈rest, owns j x
      · exact hfout x ht
      · have hp : owns pid x := by
          obtain ⟨j,hj,hjx⟩ := hx
          simp only [List.mem_cons] at hj
          rcases hj with rfl|hj
          · exact hjx
          · exact False.elim (ht ⟨j,hj,hjx⟩)
        rw [hfframe output (addr x) (Or.inr ?_)]
        · exact hmout x hp
        · intro y hy heq
          have hyx := hinj heq.symm
          subst y
          exact ht hy
    · intro r o h
      have ht : r≠output ∨ ∀ x, (∃ j∈rest, owns j x) → o≠addr x :=
        h.imp id (fun h x hx => h x (by obtain ⟨j,hj,hjx⟩ := hx; exact ⟨j,by simp [hj],hjx⟩))
      have hp : r≠output ∨ ∀ x, owns pid x → o≠addr x :=
        h.imp id (fun h x hx => h x ⟨pid,by simp,hx⟩)
      exact (hfframe r o ht).trans (hmframe r o hp)

#axiomsClean compose
end
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.GridExecution
