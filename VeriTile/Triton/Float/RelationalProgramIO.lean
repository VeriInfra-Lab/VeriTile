import VeriTile.Triton.Float.ProgramIO

/-! Logical output relations for the existing fixed Structural source schedule.

Some independent specifications relate several outputs, such as an attained
maximum and its least index, without choosing an implementation of the answer.
The witness below consists only of typed logical output values. It is bound to
the actual successful source result; safety, input domains and exact frames
are the same obligations as `ProgramIO.ImplementsOn`.
-/
namespace VeriTile.Triton.FP.Structural.ProgramIO
noncomputable section

def SatisfiesOn {α β : Type} [Inhabited β] (io : ProgramIO α β)
    (domain : io.Inputs → Prop) (predicate : io.Inputs → io.Outputs → Prop) : Prop :=
  ∀ vals, domain vals → ∀ bounds, io.InBounds bounds vals → ∀ initial,
    io.InputBinding vals initial →
    ScheduleSafe bounds (io.schedule vals) initial ∧
    ∃ final outputs,
      runSchedule (io.schedule vals) initial = some final ∧
      io.OutputBinding vals outputs final ∧ predicate vals outputs ∧
      io.Frame vals initial final

def Satisfies {α β : Type} [Inhabited β] (io : ProgramIO α β)
    (predicate : io.Inputs → io.Outputs → Prop) : Prop :=
  io.SatisfiesOn (fun _ => True) predicate

theorem SatisfiesOn.intro {α β : Type} [Inhabited β] {io : ProgramIO α β}
    {domain : io.Inputs → Prop} {predicate : io.Inputs → io.Outputs → Prop}
    (safe : ∀ vals, domain vals → ∀ bounds, io.InBounds bounds vals → ∀ initial,
      io.InputBinding vals initial → ScheduleSafe bounds (io.schedule vals) initial)
    (correct : ∀ vals, domain vals → ∀ initial, io.InputBinding vals initial →
      ∃ final outputs,
        runSchedule (io.schedule vals) initial = some final ∧
        io.OutputBinding vals outputs final ∧ predicate vals outputs ∧
        io.Frame vals initial final) : io.SatisfiesOn domain predicate := by
  intro vals admitted bounds backed initial inputs
  exact ⟨safe vals admitted bounds backed initial inputs,
    correct vals admitted initial inputs⟩

theorem Satisfies.intro {α β : Type} [Inhabited β] {io : ProgramIO α β}
    {predicate : io.Inputs → io.Outputs → Prop}
    (safe : ∀ vals bounds, io.InBounds bounds vals → ∀ initial,
      io.InputBinding vals initial → ScheduleSafe bounds (io.schedule vals) initial)
    (correct : ∀ vals initial, io.InputBinding vals initial →
      ∃ final outputs,
        runSchedule (io.schedule vals) initial = some final ∧
        io.OutputBinding vals outputs final ∧ predicate vals outputs ∧
        io.Frame vals initial final) : io.Satisfies predicate :=
  SatisfiesOn.intro (fun vals _ => safe vals) (fun vals _ => correct vals)

/-- A functional target satisfying the logical relation supplies its witness. -/
theorem SatisfiesOn.of_implementsOn {α β : Type} [Inhabited β]
    {io : ProgramIO α β} {domain : io.Inputs → Prop} {f : io.Inputs → io.Outputs}
    {predicate : io.Inputs → io.Outputs → Prop}
    (implemented : io.ImplementsOn domain f)
    (related : ∀ vals, domain vals → predicate vals (f vals)) :
    io.SatisfiesOn domain predicate := by
  intro vals admitted bounds backed initial inputs
  obtain ⟨safe, final, ran, outputs, frame⟩ :=
    implemented vals admitted bounds backed initial inputs
  exact ⟨safe, final, f vals, ran, outputs, related vals admitted, frame⟩

theorem Satisfies.of_implements {α β : Type} [Inhabited β]
    {io : ProgramIO α β} {f : io.Inputs → io.Outputs}
    {predicate : io.Inputs → io.Outputs → Prop}
    (implemented : io.Implements f)
    (related : ∀ vals, predicate vals (f vals)) : io.Satisfies predicate :=
  SatisfiesOn.of_implementsOn implemented (fun vals _ => related vals)

end
end VeriTile.Triton.FP.Structural.ProgramIO
