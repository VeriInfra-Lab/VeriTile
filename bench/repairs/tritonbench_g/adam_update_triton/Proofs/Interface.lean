import bench.repairs.tritonbench_g.adam_update_triton.IO
import bench.repairs.tritonbench_g.adam_update_triton.Proofs.AccessSafety

namespace VeriTile.Bench.Repairs.AdamUpdateTriton.Interface
open VeriTile.Triton FP.Structural
noncomputable section

theorem initial_eq {α : Type} [Inhabited α] (sem : Tile.Semantics α)
    (params : Scalars α) (total n B pid : Nat) (s : State α) :
    (invocation sem params total n B pid).initial s = Tile.programState params pid s := by
  rfl

/-- Every exterior IO cell is exterior to both source stores. -/
theorem outside_source {α β : Type} (sem : Tile.Semantics β) (params : Scalars β)
    (encode : α → β) (observe : β → α) (total n B pid : Nat)
    (values : Inputs α B) (r : RegionName) (offset : Nat)
    (outside : (programIO sem params encode observe total n B pid).Outside values r offset) :
    Tile.Outside n B pid r offset := by
  intro output
  by_cases hr : r = Tile.outputRegion output
  · right
    intro j active
    cases output
    · have h := outside.1 ⟨0, by simp [programIO, memory, inputs, outputs]⟩ j active
      exact h.resolve_left (by simpa [Tile.outputRegion] using not_ne_iff.mpr hr)
    · have h := outside.1 ⟨1, by simp [programIO, memory, inputs, outputs]⟩ j active
      exact h.resolve_left (by simpa [Tile.outputRegion] using not_ne_iff.mpr hr)
  · exact Or.inl hr

/-- Fixed-source IO assembly. The operational lemma supplies independent
lane values; source trace safety and the IO frame are proved here. -/
theorem implements {α β : Type} [Inhabited β]
    (sem : Tile.Semantics β) (params : Scalars β) (encode : α → β) (observe : β → α)
    (total n B pid : Nat) (target : Inputs α B → Output → Fin B → α)
    (correct : ∀ (values : Inputs α B) (s : State β),
      (∀ j : Fin B, pid * B + j.val < n →
        (s.mem "P" (pid * B + j.val)).read .real = encode (parameter values j)) →
      (∀ j : Fin B, pid * B + j.val < n →
        (s.mem "G" (pid * B + j.val)).read .real = encode (gradient values j)) →
      (∀ j : Fin B, pid * B + j.val < n →
        (s.mem "M" (pid * B + j.val)).read .real = encode (momentum values j)) →
      ∃ final,
        run sem.backend (source total n B).surfaceBody (Tile.programState params pid s) = some final ∧
        (∀ output (j : Fin B), pid * B + j.val < n →
          mapCell observe (final.mem (Tile.outputRegion output) (pid * B + j.val)) =
            Cell.mk .real (target values output j)) ∧
        (∀ r offset, Tile.Outside n B pid r offset → final.mem r offset = s.mem r offset)) :
    (programIO sem params encode observe total n B pid).Implements
      (fun values => result (target values .parameter) (target values .momentum)) := by
  apply ProgramIO.Implements.intro
  · intro values bounds hb s _
    rw [show (programIO sem params encode observe total n B pid).schedule values =
      [invocation sem params total n B pid] from rfl, scheduleSafe_singleton]
    change FP.Structural.TraceSafe sem.backend bounds (source total n B)
      ((invocation sem params total n B pid).initial s)
    rw [initial_eq]
    apply Tile.source_traceSafe
    · intro j active
      exact hb.2.1 ⟨0, by simp [programIO, memory, inputs, outputs]⟩ (j, PUnit.unit) active
    · intro j active
      exact hb.2.1 ⟨1, by simp [programIO, memory, inputs, outputs]⟩ (j, PUnit.unit) active
    · intro j active
      exact hb.2.1 ⟨2, by simp [programIO, memory, inputs, outputs]⟩ (j, PUnit.unit) active
  · intro values s hin
    obtain ⟨final, ran, valuesOut, frame⟩ := correct values s
      (fun j active => hin ⟨0, by simp [programIO, memory, inputs, outputs]⟩ (j, PUnit.unit) active)
      (fun j active => hin ⟨1, by simp [programIO, memory, inputs, outputs]⟩ (j, PUnit.unit) active)
      (fun j active => hin ⟨2, by simp [programIO, memory, inputs, outputs]⟩ (j, PUnit.unit) active)
    refine ⟨final, ?_, ?_, ?_⟩
    · change runSchedule [invocation sem params total n B pid] s = some final
      rw [runSchedule_singleton]
      change run sem.backend (source total n B).surfaceBody
        ((invocation sem params total n B pid).initial s) = some final
      simpa only [initial_eq] using ran
    · intro o j active
      fin_cases o
      · exact valuesOut .parameter j.1 active
      · exact valuesOut .momentum j.1 active
    · intro r offset outside
      exact frame r offset (outside_source sem params encode observe total n B pid values r offset outside)

end
end VeriTile.Bench.Repairs.AdamUpdateTriton.Interface
