import bench.tritonbench_g.fast_rms_layernorm.Kernels

/-! Finite grid composition permits overlapping immutable reads. Only writes
must be separated from the other program's reads and writes. Public entries
instantiate the step premise with complete source proofs. -/
namespace VeriTile.Bench.Repairs.FastRmsLayernorm.GridRun
noncomputable section
variable {W A β γ : Type}
abbrev Heap (A β : Type) := A → β

def run (execute : W → Heap A β → Option (Heap A β)) : List W → Heap A β → Option (Heap A β)
  | [],u => some u
  | w::ws,u => (execute w u).bind (run execute ws)

theorem complete (execute : W → Heap A β → Option (Heap A β))
    (I : W → Type) (address : (w : W) → I w → A) (read : β → γ)
    (expected : Heap A β → (w : W) → I w → γ) (reads writes : W → A → Prop)
    (inside : ∀ w i,writes w (address w i))
    (jobs : List W) (unique : jobs.Nodup)
    (separate : ∀ a∈jobs, ∀ b∈jobs,a≠b → ∀ addr,writes a addr → ¬writes b addr ∧ ¬reads b addr)
    (step : ∀ w∈jobs, ∀ u,∃ v,execute w u=some v ∧
      (∀ i,read (v (address w i))=expected u w i) ∧
      (∀ addr,¬writes w addr → v addr=u addr))
    (stable : ∀ w∈jobs, ∀ u v,(∀ addr,reads w addr → u addr=v addr) →
      ∀ i,expected u w i=expected v w i)
    (u : Heap A β) :
    ∃ v,run execute jobs u=some v ∧
      (∀ w∈jobs, ∀ i,read (v (address w i))=expected u w i) ∧
      (∀ addr,(∀ w∈jobs,¬writes w addr) → v addr=u addr) := by
  induction jobs generalizing u with
  | nil => exact ⟨u,rfl,by simp,by intros; rfl⟩
  | cons w ws ih =>
    obtain ⟨v,he,out,frame⟩ := step w (by simp) u
    obtain ⟨final,hfinal,outs,frames⟩ := ih unique.tail
      (fun a ha b hb => separate a (List.mem_cons_of_mem _ ha) b (List.mem_cons_of_mem _ hb))
      (fun a ha => step a (List.mem_cons_of_mem _ ha))
      (fun a ha => stable a (List.mem_cons_of_mem _ ha)) v
    refine ⟨final,by simp [run,he,hfinal],?_,?_⟩
    · intro a member i
      rcases List.mem_cons.mp member with same|member
      · subst a
        rw [frames]
        · exact out i
        · intro b hb
          exact (separate w (by simp) b (List.mem_cons_of_mem _ hb)
            (fun eq => (List.nodup_cons.mp unique).1 (eq.symm ▸ hb)) _ (inside w i)).1
      · rw [outs a member i]
        apply stable a (List.mem_cons_of_mem _ member) v u
        intro addr ha
        apply frame
        intro written
        exact (separate w (by simp) a (List.mem_cons_of_mem _ member)
          (fun eq => (List.nodup_cons.mp unique).1 (eq.symm ▸ member)) addr written).2 ha
    · intro addr outside
      rw [frames addr (fun a ha => outside a (List.mem_cons_of_mem _ ha))]
      exact frame addr (outside w (by simp))

#axiomsClean complete
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.GridRun
