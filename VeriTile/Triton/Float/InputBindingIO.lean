import VeriTile.Triton.Float.ProgramEquivalence

/-! Explicit exact-cell input binding for existing typed source proofs.

`ProgramIO.InputBinding` deliberately pins decoded reads, including the
interpreter's default when a stored tag differs. Some existing source proofs
instead require the active input cells to have their declared tags and values.
This additive interface states that stronger premise explicitly. It retains
the same source schedules, access safety, logical output binding and raw frame;
it does not change `ProgramIO` or add a custom execution/predicate callback.
-/
namespace VeriTile.Triton.FP.Structural
namespace ProgramIO
noncomputable section

/-- Every active input has exactly the declared dtype and encoded payload.
Inactive cells remain unrestricted. Discrete metadata is preserved by
`mapValue`, just as in the decoded-read interface. -/
def InputCells {α β : Type} (io : ProgramIO α β)
    (vals : io.Inputs) (initial : State β) : Prop :=
  ∀ i j, io.inputMask i vals j →
    initial.mem io.inputs[i].buf (io.read i vals j) =
      Cell.mk io.inputs[i].dtype (mapValue io.encode io.inputs[i].dtype (vals i j))

/-- Exact active input cells imply the original decoded-read binding.
The converse is intentionally absent: a mismatched tag can decode to default.
-/
theorem InputCells.toInputBinding {α β : Type} [Inhabited β]
    {io : ProgramIO α β} {vals : io.Inputs} {initial : State β}
    (bound : io.InputCells vals initial) : io.InputBinding vals initial := by
  intro i j active
  rw [bound i j active]
  simp [Cell.read]

/-- Source correctness with an explicit exact-cell input premise. The domain
contains only logical conditions; access safety and execution remain results.
-/
def ImplementsOnCells {α β : Type} [Inhabited β] (io : ProgramIO α β)
    (domain : io.Inputs → Prop) (f : io.Inputs → io.Outputs) : Prop :=
  ∀ vals, domain vals → ∀ bounds, io.InBounds bounds vals → ∀ initial,
    io.InputCells vals initial →
    ScheduleSafe bounds (io.schedule vals) initial ∧
    ∃ final, runSchedule (io.schedule vals) initial = some final ∧
      io.OutputBinding vals (f vals) final ∧ io.Frame vals initial final

def ImplementsCells {α β : Type} [Inhabited β] (io : ProgramIO α β)
    (f : io.Inputs → io.Outputs) : Prop := io.ImplementsOnCells (fun _ => True) f

theorem ImplementsOnCells.intro {α β : Type} [Inhabited β] {io : ProgramIO α β}
    {domain : io.Inputs → Prop} {f : io.Inputs → io.Outputs}
    (safe : ∀ vals, domain vals → ∀ bounds, io.InBounds bounds vals →
      ∀ initial, io.InputCells vals initial →
        ScheduleSafe bounds (io.schedule vals) initial)
    (correct : ∀ vals, domain vals → ∀ bounds, io.InBounds bounds vals →
      ∀ initial, io.InputCells vals initial →
        ∃ final, runSchedule (io.schedule vals) initial = some final ∧
          io.OutputBinding vals (f vals) final ∧ io.Frame vals initial final) :
    io.ImplementsOnCells domain f := by
  intro vals domain bounds bounded initial inputs
  exact ⟨safe vals domain bounds bounded initial inputs,
    correct vals domain bounds bounded initial inputs⟩

theorem ImplementsCells.intro {α β : Type} [Inhabited β] {io : ProgramIO α β}
    {f : io.Inputs → io.Outputs}
    (safe : ∀ vals bounds, io.InBounds bounds vals → ∀ initial,
      io.InputCells vals initial → ScheduleSafe bounds (io.schedule vals) initial)
    (correct : ∀ vals initial, io.InputCells vals initial →
      ∃ final, runSchedule (io.schedule vals) initial = some final ∧
        io.OutputBinding vals (f vals) final ∧ io.Frame vals initial final) :
    io.ImplementsCells f := by
  intro vals _ bounds bounded initial inputs
  exact ⟨safe vals bounds bounded initial inputs, correct vals initial inputs⟩

/-- A decoded-read proof also works on exact cells, with no new source proof. -/
theorem ImplementsOn.toCells {α β : Type} [Inhabited β] {io : ProgramIO α β}
    {domain : io.Inputs → Prop} {f : io.Inputs → io.Outputs}
    (proof : io.ImplementsOn domain f) : io.ImplementsOnCells domain f := by
  intro vals domain bounds bounded initial inputs
  exact proof vals domain bounds bounded initial inputs.toInputBinding

theorem Implements.toCells {α β : Type} [Inhabited β] {io : ProgramIO α β}
    {f : io.Inputs → io.Outputs} (proof : io.Implements f) : io.ImplementsCells f :=
  ImplementsOn.toCells proof

/-- Both actual schedules use the same explicitly exact input cells. Output
equality and both exterior frames are unchanged from `EquivalentOn`. -/
def EquivalentOnCells {α β : Type} [Inhabited β] (io : ProgramIO α β)
    (candidate : Variant io) (domain : io.Inputs → Prop) : Prop :=
  ∀ vals, domain vals → ∀ bounds, io.InBounds bounds vals → ∀ initial,
    io.InputCells vals initial →
    ScheduleSafe bounds (io.schedule vals) initial ∧
    ScheduleSafe bounds (candidate.schedule vals) initial ∧
    ∃ left right,
      runSchedule (io.schedule vals) initial = some left ∧
      runSchedule (candidate.schedule vals) initial = some right ∧
      io.OutputsEqual candidate vals left right ∧
      io.Frame vals initial left ∧ io.Frame vals initial right

def EquivalentCells {α β : Type} [Inhabited β] (io : ProgramIO α β)
    (candidate : Variant io) : Prop := io.EquivalentOnCells candidate (fun _ => True)

theorem EquivalentOnCells.intro {α β : Type} [Inhabited β] {io : ProgramIO α β}
    {candidate : Variant io} {domain : io.Inputs → Prop}
    (leftSafe : ∀ vals, domain vals → ∀ bounds, io.InBounds bounds vals →
      ∀ initial, io.InputCells vals initial →
        ScheduleSafe bounds (io.schedule vals) initial)
    (rightSafe : ∀ vals, domain vals → ∀ bounds, io.InBounds bounds vals →
      ∀ initial, io.InputCells vals initial →
        ScheduleSafe bounds (candidate.schedule vals) initial)
    (runs : ∀ vals, domain vals → ∀ bounds, io.InBounds bounds vals →
      ∀ initial, io.InputCells vals initial →
        ∃ left right,
          runSchedule (io.schedule vals) initial = some left ∧
          runSchedule (candidate.schedule vals) initial = some right ∧
          io.OutputsEqual candidate vals left right ∧
          io.Frame vals initial left ∧ io.Frame vals initial right) :
    io.EquivalentOnCells candidate domain := by
  intro vals domain bounds bounded initial inputs
  exact ⟨leftSafe vals domain bounds bounded initial inputs,
    rightSafe vals domain bounds bounded initial inputs,
    runs vals domain bounds bounded initial inputs⟩

theorem EquivalentOn.toCells {α β : Type} [Inhabited β] {io : ProgramIO α β}
    {candidate : Variant io} {domain : io.Inputs → Prop}
    (proof : io.EquivalentOn candidate domain) : io.EquivalentOnCells candidate domain := by
  intro vals domain bounds bounded initial inputs
  exact proof vals domain bounds bounded initial inputs.toInputBinding

theorem Equivalent.toCells {α β : Type} [Inhabited β] {io : ProgramIO α β}
    {candidate : Variant io} (proof : io.Equivalent candidate) : io.EquivalentCells candidate :=
  EquivalentOn.toCells proof

/-- Matching logical results of two exact-cell implementations imply their
observed equivalence. Successful execution is still required on both sides. -/
theorem EquivalentOnCells.of_implementsOnCells {α β : Type} [Inhabited β]
    {io : ProgramIO α β} {candidate : Variant io}
    {domain : io.Inputs → Prop} {f : io.Inputs → io.Outputs}
    (left : io.ImplementsOnCells domain f)
    (right : candidate.toProgramIO.ImplementsOnCells domain f) :
    io.EquivalentOnCells candidate domain := by
  intro vals domain bounds bounded initial inputs
  obtain ⟨leftSafe, l, leftRun, leftOut, leftFrame⟩ := left vals domain bounds bounded initial inputs
  obtain ⟨rightSafe, r, rightRun, rightOut, rightFrame⟩ := right vals domain bounds bounded initial inputs
  refine ⟨leftSafe, rightSafe, l, r, leftRun, rightRun, ?_, leftFrame, rightFrame⟩
  intro o j active
  exact (leftOut o j active).trans (rightOut o j active).symm

end
end ProgramIO
end VeriTile.Triton.FP.Structural
