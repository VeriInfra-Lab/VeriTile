import bench.tritonbench_g.nested_loops_processing.IO
import bench.tritonbench_g.nested_loops_processing.Proofs.SourceSafety
import bench.tritonbench_g.nested_loops_processing.Proofs.KernelState

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.IOProof
open VeriTile.Triton FP.Structural KernelResult TypedExecution
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000

theorem invocation_initial {α : Type} (M : Algebra α) (candidate : Bool)
    (rows cols : Nat) (s : State α) :
    (IO.invocation M candidate rows cols).initial s = s := by
  cases s
  rfl

/-- Full output-cell result, including all cells untouched by the stores. -/
theorem implements {α : Type} [Inhabited α] (M : Algebra α) (candidate : Bool)
    (rows cols : Nat) (positive : 0 < rows*cols) :
    (IO.nestedIO M candidate rows cols).ImplementsOnCells (IO.initialized rows cols)
      (IO.result rows cols (fun i => signedValue (wordTarget rows cols i))) := by
  apply ProgramIO.ImplementsOnCells.intro
  · intro vals domain bounds bounded s binding
    rw [scheduleSafe_singleton]
    change TraceSafe M bounds (Kernels.kernel candidate "input" "output" (rows*cols) cols)
      ((IO.invocation M candidate rows cols).initial s)
    apply SourceSafety.source_safe M candidate (rows*cols) cols positive
    · have h := bounded.2.1 0 (⟨rows*cols-1,by omega⟩,()) trivial
      change rows*cols-1 < bounds "input" at h
      omega
    · have h := bounded.2.1 1 (⟨rows*cols-1,by omega⟩,()) trivial
      change rows*cols-1 < bounds "output" at h
      omega
  · intro vals domain bounds bounded s binding
    have input : ∀ i, i < rows*cols → s.mem "input" i = .mk .int (signedValue (arangeWord i)) := by
      intro i hi
      have h := binding 0 (⟨i,hi⟩,()) trivial
      change s.mem "input" i = .mk .int (vals 0 (⟨i,hi⟩,())) at h
      exact h.trans (congrArg (Cell.mk .int) (domain.1 (⟨i,hi⟩,())))
    have output : ∀ i, i < rows*cols → s.mem "output" i = .mk .int (0 : Int) := by
      intro i hi
      have h := binding 1 (⟨i,hi⟩,()) trivial
      change s.mem "output" i = .mk .int (vals 1 (⟨i,hi⟩,())) at h
      exact h.trans (congrArg (Cell.mk .int) (domain.2 (⟨i,hi⟩,())))
    refine ⟨finalState (rows*cols) cols s,?_,?_,?_⟩
    · rw [runSchedule_singleton]
      change exec M (Kernels.kernel candidate "input" "output" (rows*cols) cols)
        ((IO.invocation M candidate rows cols).initial s) = _
      rw [invocation_initial]
      cases candidate
      · exact source_run M (rows*cols) cols s
      · exact candidate_run M (rows*cols) cols s
    · intro o j _
      fin_cases o
      change mapCell id ((finalState (rows*cols) cols s).mem "output" j.1.val) =
        Cell.mk .int (signedValue (wordTarget rows cols j.1.val))
      rw [initialized_cell s rows cols positive input output j.1.val j.1.isLt]
      rfl
    · intro r i outside
      apply final_frame
      by_cases same : r = "output"
      · right
        by_contra hn
        have hi : i < rows*cols := by omega
        rcases outside.1 0 (⟨i,hi⟩,()) trivial with h | h
        · exact h same
        · exact h rfl
      · exact Or.inl same

theorem equivalent {α : Type} [Inhabited α] (M : Algebra α)
    (rows cols : Nat) (positive : 0 < rows*cols) :
    (IO.nestedIO M Bool.false rows cols).EquivalentOnCells (IO.commuted M rows cols)
      (IO.initialized rows cols) :=
  ProgramIO.EquivalentOnCells.of_implementsOnCells
    (implements M Bool.false rows cols positive) (implements M Bool.true rows cols positive)

#axiomsClean implements
#axiomsClean equivalent
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.NestedLoopsProcessing.IOProof
