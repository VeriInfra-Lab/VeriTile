import bench.tritonbench_g.triton_conv2d_fwd.Proofs.KernelGrid
namespace VeriTile.Bench.Repairs.Conv2d.KernelGrid
open VeriTile.Triton FP.Structural KernelBinding
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 8192
variable {α : Type} [Inhabited α]

theorem program_memory (M : Algebra α) (H : KernelHalf.Dot α) (D : KernelDefault.Dot α)
    (s : Shape) (v : Valid s) (f : Flags) (pid : Program) (hg : pid.2.2<s.groups) (t : State α) :
    ∃ u,program M H D s f pid t=some u ∧ SameInputs u t ∧
      (∀ x,s.Active x → u.mem "output" (outputPointer s x)=
        if owner x=pid then .mk .fp32 (point M H D s f t x) else t.mem "output" (outputPointer s x)) ∧
      (∀ (r : RegionName) o,r≠"output" ∨ s.size≤o → u.mem r o=t.mem r o) := by
  obtain ⟨u,he,hr,hf⟩:=program_correct M H D s v f pid hg t
  refine ⟨u,he,⟨?_,?_⟩,?_,?_⟩
  · intro o; exact hf "input" o (Or.inl (by decide))
  · intro o; exact hf "weight" o (Or.inl (by decide))
  · intro x hx
    by_cases ho : owner x=pid
    · subst pid
      simpa only [cell_index,point,if_pos rfl] using hr (index x) (by simpa only [cell_index] using hx)
    · rw [if_neg ho]
      apply hf _ _ (Or.inr ?_)
      intro i hi eq
      have same:=address_injective s v x (cell pid i) hx hi eq
      have own:=congrArg owner same
      rw [owner_cell] at own
      exact ho own
  · intro r o outside
    apply hf r o
    rcases outside with outside|outside
    · exact Or.inl outside
    · exact Or.inr (fun i hi => Nat.ne_of_gt (lt_of_lt_of_le (address_bound s v _ hi) outside))

theorem run_correct (M : Algebra α) (H : KernelHalf.Dot α) (D : KernelDefault.Dot α)
    (s : Shape) (v : Valid s) (f : Flags) (ps : List Program)
    (hg : ∀ pid∈ps,pid.2.2<s.groups) (t : State α) :
    ∃ u,run M H D s f ps t=some u ∧ SameInputs u t ∧
      (∀ x,s.Active x → u.mem "output" (outputPointer s x)=
        if owner x∈ps then .mk .fp32 (point M H D s f t x) else t.mem "output" (outputPointer s x)) ∧
      (∀ (r : RegionName) o,r≠"output" ∨ s.size≤o → u.mem r o=t.mem r o) := by
  induction ps generalizing t with
  | nil =>
    exact ⟨t,rfl,⟨by intros;rfl,by intros;rfl⟩,by intros;simp,by intros;rfl⟩
  | cons pid ps ih =>
    obtain ⟨mid,hm,mi,mr,mf⟩:=program_memory M H D s v f pid (hg pid (by simp)) t
    obtain ⟨u,hu,ui,ur,uf⟩:=ih (fun q hq=>hg q (by simp [hq])) mid
    refine ⟨u,?_,⟨?_,?_⟩,?_,?_⟩
    · simp only [run,hm,Option.bind_some,hu]
    · intro o; exact (ui.1 o).trans (mi.1 o)
    · intro o; exact (ui.2 o).trans (mi.2 o)
    · intro x hx
      rw [ur x hx,point_inputs M H D s f mid t mi x]
      by_cases hp : owner x∈ps
      · simp [hp]
      · rw [if_neg hp,mr x hx]
        simp [hp]
    · intro r o outside
      exact (uf r o outside).trans (mf r o outside)

theorem owner_mem (s : Shape) (v : Valid s) (x : Conv2d.Cell) (hx : s.Active x) :
    owner x∈programs s := by
  obtain ⟨pid,hp,i,rfl⟩:=covers s v x hx
  simpa only [owner_cell] using hp

/-- Complete original grid, including empty batch and redundant masked CTAs.
The result is obtained by successful shared-DSL execution of every program on
the evolving state. Input-snapshot equality and the exterior frame are derived. -/
theorem launch_correct (M : Algebra α) (H : KernelHalf.Dot α) (D : KernelDefault.Dot α)
    (s : Shape) (v : Valid s) (f : Flags) (t : State α) :
    ∃ u,launch M H D s f t=some u ∧ SameInputs u t ∧
      (∀ x,s.Active x → u.mem "output" (outputPointer s x)=.mk .fp32 (point M H D s f t x)) ∧
      (∀ (r : RegionName) o,r≠"output" ∨ s.size≤o → u.mem r o=t.mem r o) := by
  have hg : ∀ pid∈programs s,pid.2.2<s.groups := by
    intro pid hp
    rcases pid with ⟨a,b,g⟩
    simp only [programs,List.mem_product,List.mem_range] at hp
    exact hp.2.2
  obtain ⟨u,he,hi,hr,hf⟩:=run_correct M H D s v f (programs s) hg t
  exact ⟨u,he,hi,fun x hx => by simpa only [if_pos (owner_mem s v x hx)] using hr x hx,hf⟩

#axiomsClean program_memory
#axiomsClean run_correct
#axiomsClean launch_correct
end
end VeriTile.Bench.Repairs.Conv2d.KernelGrid
