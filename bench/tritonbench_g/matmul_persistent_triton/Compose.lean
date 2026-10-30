import bench.tritonbench_g.int_scaled_matmul.Sources.GridExecution

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Compose
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.GridExecution (Outside Post)
noncomputable section
set_option maxHeartbeats 5000000

/-- Compose actual body executions while preserving the control registers
proved from the quote prefix. The final theorem discharges this invariant. -/
theorem ready {α I : Type} (output : RegionName) (addr : I → Nat)
    (hinj : Function.Injective addr) (value : I → Cell α) (owns : Nat → I → Prop)
    (one : Nat → State α → Option (State α)) (Ready : State α → Prop)
    (initial : State α) (count : Nat)
    (single : ∀ pid,pid<count → ∀ current,Outside output initial current → Ready current →
      ∃ final,one pid current=some final ∧ Post output addr value (owns pid) current final ∧ Ready final)
    (programs : List Nat) (hprograms : ∀ pid∈programs,pid<count)
    (current : State α) (hout : Outside output initial current) (hready : Ready current) :
    ∃ final,VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.GridExecution.sequence one programs current=some final ∧
      Post output addr value (fun x => ∃ pid∈programs,owns pid x) current final ∧ Ready final := by
  classical
  induction programs generalizing current with
  | nil =>
    refine ⟨current,rfl,⟨?_,?_⟩,hready⟩
    · simp
    · intros; rfl
  | cons pid rest ih =>
    obtain ⟨mid,hm,⟨hmout,hmframe⟩,hmready⟩ := single pid (hprograms pid (by simp)) current hout hready
    have midOutside : Outside output initial mid := by
      intro r o h
      exact (hmframe r o (.inl h)).trans (hout r o h)
    obtain ⟨final,hf,⟨hfout,hfframe⟩,hfready⟩ := ih (fun j hj => hprograms j (by simp [hj])) mid midOutside hmready
    refine ⟨final,?_,⟨?_,?_⟩,hfready⟩
    · simp only [VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.GridExecution.sequence,hm,Option.bind_some,hf]
    · intro x hx
      by_cases tail : ∃ j∈rest,owns j x
      · exact hfout x tail
      · have head : owns pid x := by
          obtain ⟨j,hj,hjx⟩ := hx
          simp only [List.mem_cons] at hj
          rcases hj with rfl|hj
          · exact hjx
          · exact False.elim (tail ⟨j,hj,hjx⟩)
        rw [hfframe output (addr x) (.inr ?_)]
        · exact hmout x head
        · intro y hy equal
          have same := hinj equal.symm
          subst y
          exact tail hy
    · intro r o h
      have tail : r≠output ∨ ∀ x,(∃ j∈rest,owns j x) → o≠addr x :=
        h.imp id (fun h x hx => h x (by obtain ⟨j,hj,hjx⟩ := hx; exact ⟨j,by simp [hj],hjx⟩))
      have head : r≠output ∨ ∀ x,owns pid x → o≠addr x :=
        h.imp id (fun h x hx => h x ⟨pid,by simp,hx⟩)
      exact (hfframe r o tail).trans (hmframe r o head)

def ordinal {α : Type} [Inhabited α] (M : Algebra α) (name : RegName)
    (body : List ComputeStmt) (i : Nat) (s : State α) : Option (State α) :=
  run M body (s.setReg name .nat [] (fun _ => i))

theorem range_sequence {α : Type} [Inhabited α] (M : Algebra α) (name : RegName)
    (body : List ComputeStmt) (start stop : Nat) (s : State α) :
    FP.Structural.range M name start stop 1 body s=
      VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.GridExecution.sequence (ordinal M name body) (List.range' start (stop-start)) s := by
  have aux : ∀ fuel start (s : State α),stop-start=fuel →
      FP.Structural.range M name start stop 1 body s=
        VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.GridExecution.sequence (ordinal M name body) (List.range' start fuel) s := by
    intro fuel
    induction fuel with
    | zero =>
      intro start s he
      rw [FP.Structural.range,if_neg (by decide),if_neg (by omega)]
      rfl
    | succ fuel ih =>
      intro start s he
      rw [FP.Structural.range,if_neg (by decide),if_pos (by omega)]
      simp only [List.range'_succ,VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.GridExecution.sequence,ordinal]
      apply congrArg (fun f => (run M body (s.setReg name .nat [] (fun _ => start))).bind f)
      funext next
      exact ih (start+1) next (by omega)
  exact aux _ start s rfl

#axiomsClean ready
#axiomsClean range_sequence
end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Compose
