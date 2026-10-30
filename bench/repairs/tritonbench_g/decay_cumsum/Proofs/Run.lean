import bench.repairs.tritonbench_g.decay_cumsum.Proofs.Grid

/-! A general finite-grid composition rule. Reader contracts instantiate its
step premise with the proved complete source, never an execution-success
assumption. Separate program footprints include every read and every write.
-/
namespace VeriTile.Bench.Repairs.DecayCumsum.Run
noncomputable section
variable {W β γ : Type}
abbrev Heap (β : Type) := Nat → β

def run (execute : W → Heap β → Option (Heap β)) : List W → Heap β → Option (Heap β)
  | [],u => some u
  | w::ws,u => (execute w u).bind (run execute ws)

theorem complete (execute : W → Heap β → Option (Heap β))
    (I : W → Type) (address : (w : W) → I w → Nat) (read : β → γ)
    (expected : Heap β → (w : W) → I w → γ) (footprint : W → Nat → Prop)
    (inside : ∀ w i, footprint w (address w i))
    (jobs : List W) (unique : jobs.Nodup)
    (disjoint : ∀ a∈jobs, ∀ b∈jobs, a≠b → ∀ addr, footprint a addr → ¬footprint b addr)
    (step : ∀ w∈jobs, ∀ u, ∃ v, execute w u=some v ∧
      (∀ i, read (v (address w i))=expected u w i) ∧
      (∀ addr, ¬footprint w addr → v addr=u addr))
    (stable : ∀ w∈jobs, ∀ u v, (∀ addr, footprint w addr → u addr=v addr) →
      ∀ i, expected u w i=expected v w i)
    (u : Heap β) :
    ∃ v, run execute jobs u=some v ∧
      (∀ w∈jobs, ∀ i, read (v (address w i))=expected u w i) ∧
      (∀ addr, (∀ w∈jobs, ¬footprint w addr) → v addr=u addr) := by
  induction jobs generalizing u with
  | nil => exact ⟨u,rfl,by simp,by intros; rfl⟩
  | cons w ws ih =>
    obtain ⟨v,he,out,frame⟩ := step w (by simp) u
    obtain ⟨final,hfinal,outs,frames⟩ := ih unique.tail
      (fun a ha b hb => disjoint a (List.mem_cons_of_mem _ ha) b (List.mem_cons_of_mem _ hb))
      (fun a ha => step a (List.mem_cons_of_mem _ ha))
      (fun a ha => stable a (List.mem_cons_of_mem _ ha)) v
    refine ⟨final,by simp [run,he,hfinal],?_,?_⟩
    · intro a member i
      rcases List.mem_cons.mp member with same|member
      · subst a
        rw [frames]
        · exact out i
        · intro b hb
          exact disjoint w (by simp) b (List.mem_cons_of_mem _ hb)
            (fun eq => (List.nodup_cons.mp unique).1 (eq.symm ▸ hb)) _ (inside w i)
      · rw [outs a member i]
        apply stable a (List.mem_cons_of_mem _ member) v u
        intro addr ha
        apply frame
        exact disjoint a (List.mem_cons_of_mem _ member) w (by simp)
          (fun eq => (List.nodup_cons.mp unique).1 (eq ▸ member)) addr ha
    · intro addr outside
      rw [frames addr (fun a ha => outside a (List.mem_cons_of_mem _ ha))]
      exact frame addr (outside w (by simp))

#axiomsClean complete
end
end VeriTile.Bench.Repairs.DecayCumsum.Run
