import bench.repairs.tritonbench_g.swiglu_triton.IO
import bench.repairs.tritonbench_g.swiglu_triton.Proofs.LogicalExecution
import bench.repairs.tritonbench_g.swiglu_triton.Proofs.SourceSafety
import VeriTile.Triton.Float.ProgramIO

namespace VeriTile.Bench.Repairs.SwigluTriton.PublicIO
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem implements {α β : Type} [Inhabited β] (M : Algebra β)
    (encode : α → β) (observe : β → α) (f : Channel → α → α → α → α)
    (lower : ∀ ch a b dc, observe (DirectTile.scalar M ch (encode a) (encode b) (encode dc)) = f ch a b dc)
    (backward : Bool) (block cols row : Nat) :
    (io M encode observe backward block cols row).Implements (formula f block) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds bounded s binding
    rw [scheduleSafe_singleton]
    change TraceSafe M bounds (directSource backward block cols) (DirectTile.rowState row s)
    apply SourceSafety.source_safe
    · intro j active
      exact bounded.2.1 ⟨0, by simp [io, inputChannels, outputChannels, enabled]⟩ (j, PUnit.unit) active
    · intro j active
      exact bounded.2.1 ⟨1, by simp [io, inputChannels, outputChannels, enabled]⟩ (j, PUnit.unit) active
    · intro on j active
      exact bounded.2.1 ⟨2, by simp [io, inputChannels, outputChannels, enabled]⟩ (j, PUnit.unit) ⟨on, active⟩
    · intro on j active
      exact bounded.2.2.1 ⟨0, by simp [io, inputChannels, outputChannels, enabled]⟩ (j, PUnit.unit) ⟨on, active⟩
  · intro vals s binding
    have run := LogicalExecution.inputs M backward block cols row s
      (fun j => encode (vals ⟨0, by simp [io, inputChannels, outputChannels, enabled]⟩ (j, PUnit.unit)))
      (fun j => encode (vals ⟨1, by simp [io, inputChannels, outputChannels, enabled]⟩ (j, PUnit.unit)))
      (fun j => encode (vals ⟨2, by simp [io, inputChannels, outputChannels, enabled]⟩ (j, PUnit.unit)))
      (fun j h => binding ⟨0, by simp [io, inputChannels, outputChannels, enabled]⟩ (j, PUnit.unit) h)
      (fun j h => binding ⟨1, by simp [io, inputChannels, outputChannels, enabled]⟩ (j, PUnit.unit) h)
      (fun on j h => binding ⟨2, by simp [io, inputChannels, outputChannels, enabled]⟩ (j, PUnit.unit) ⟨on, h⟩)
    obtain ⟨final, executed, values, frame⟩ := run
    refine ⟨final, ?_, ?_, ?_⟩
    · simpa only [io, runSchedule_singleton, Invocation.run, Invocation.initial,
        invocation, bindRegisters, List.foldl_nil, Option.getD_none, Bool.false_eq_true,
        ↓reduceIte, DirectTile.rowState] using executed
    · intro o j active
      fin_cases o
      all_goals
        dsimp [io, outputWrite, outputMask, formula] at active ⊢
        rcases active with ⟨on, active⟩
        subst backward
      · change mapCell observe (final.mem (outputRegion .c) (row*cols+j.1.val)) =
          Cell.mk .real (f .c (vals 0 j) (vals 1 j) (vals 2 j))
        rw [values .c rfl j.1 active]
        exact congrArg (Cell.mk .real) (lower .c _ _ _)
      · change mapCell observe (final.mem (outputRegion .da) (row*cols+j.1.val)) =
          Cell.mk .real (f .da (vals 0 j) (vals 1 j) (vals 2 j))
        rw [values .da (by simp [io, inputChannels, outputChannels, enabled]) j.1 active]
        exact congrArg (Cell.mk .real) (lower .da _ _ _)
      · change mapCell observe (final.mem (outputRegion .db) (row*cols+j.1.val)) =
          Cell.mk .real (f .db (vals 0 j) (vals 1 j) (vals 2 j))
        rw [values .db (by simp [io, inputChannels, outputChannels, enabled]) j.1 active]
        exact congrArg (Cell.mk .real) (lower .db _ _ _)
    · intro r offset outside
      apply frame
      intro ch on
      by_cases same : r = outputRegion ch
      · right
        intro j active equal
        cases backward <;> cases ch <;> simp only [enabled] at on
        · rcases outside.1 ⟨0, by simp [io]⟩ j ⟨rfl, active⟩ with h | h
          · exact h same
          · exact h equal
        · contradiction
        · contradiction
        · contradiction
        · rcases outside.1 ⟨1, by simp [io]⟩ j ⟨rfl, active⟩ with h | h
          · exact h same
          · exact h equal
        · rcases outside.1 ⟨2, by simp [io]⟩ j ⟨rfl, active⟩ with h | h
          · exact h same
          · exact h equal
      · exact Or.inl same

#axiomsClean implements
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluTriton.PublicIO
