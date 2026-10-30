import VeriTile.Triton.Float.ProgramIO

/-! Equivalence of two actual structural launch schedules on one typed IO layout.

Both schedules use the shared input encoding, channels, windows, masks and
scratch footprint. A variant changes only its concrete source invocations and
its declared native observation. The judgment requires both actual executions
to succeed, both source traces to be safe, active observed output cells to agree,
and both complete exterior cell frames. It introduces no numerical rewrite law.
-/
namespace VeriTile.Triton.FP.Structural
namespace ProgramIO
noncomputable section

/-- Another implementation of the same public typed memory interface.
The schedule still consists only of actual `Invocation`s; there is no executor
or correctness-predicate callback. The observer preserves cell dtype/discrete
payloads through the shared `mapCell`. -/
structure Variant {α β : Type} (io : ProgramIO α β) where
  schedule : io.Inputs → List (Invocation β)
  observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → β → α

/-- Reuse exactly the original channels, input encoding and memory footprint. -/
def Variant.toProgramIO {α β : Type} {io : ProgramIO α β}
    (candidate : Variant io) : ProgramIO α β :=
  { io with schedule := candidate.schedule, observe := candidate.observe }

/-- Complete active output cells agree after their declared native observation.
Equality preserves dtype tags and all discrete payloads, not just float reads. -/
def OutputsEqual {α β : Type} (io : ProgramIO α β) (candidate : Variant io)
    (vals : io.Inputs) (left right : State β) : Prop :=
  ∀ o j, io.outputMask o vals j →
    mapCell (io.observe o j) (left.mem io.outputs[o].buf (io.write o vals j)) =
      mapCell (candidate.observe o j) (right.mem io.outputs[o].buf (io.write o vals j))

/-- Actual two-schedule equivalence under an explicit, independent logical
input domain. Trace safety and successful runs are conclusions; identical
failing programs therefore do not satisfy this judgment automatically. -/
def EquivalentOn {α β : Type} [Inhabited β] (io : ProgramIO α β)
    (candidate : Variant io) (domain : io.Inputs → Prop) : Prop :=
  ∀ vals, domain vals → ∀ bounds, io.InBounds bounds vals → ∀ initial,
    io.InputBinding vals initial →
    ScheduleSafe bounds (io.schedule vals) initial ∧
    ScheduleSafe bounds (candidate.schedule vals) initial ∧
    ∃ left right,
      runSchedule (io.schedule vals) initial = some left ∧
      runSchedule (candidate.schedule vals) initial = some right ∧
      io.OutputsEqual candidate vals left right ∧
      io.Frame vals initial left ∧ io.Frame vals initial right

def Equivalent {α β : Type} [Inhabited β] (io : ProgramIO α β)
    (candidate : Variant io) : Prop := io.EquivalentOn candidate (fun _ => True)

/-- Assemble independent safety and complete execution/equivalence proofs.
These are proof obligations of the shared judgment, never fields of an IO. -/
theorem EquivalentOn.intro {α β : Type} [Inhabited β] {io : ProgramIO α β}
    {candidate : Variant io} {domain : io.Inputs → Prop}
    (leftSafe : ∀ vals, domain vals → ∀ bounds, io.InBounds bounds vals →
      ∀ initial, io.InputBinding vals initial →
        ScheduleSafe bounds (io.schedule vals) initial)
    (rightSafe : ∀ vals, domain vals → ∀ bounds, io.InBounds bounds vals →
      ∀ initial, io.InputBinding vals initial →
        ScheduleSafe bounds (candidate.schedule vals) initial)
    (runs : ∀ vals, domain vals → ∀ bounds, io.InBounds bounds vals →
      ∀ initial, io.InputBinding vals initial →
        ∃ left right,
          runSchedule (io.schedule vals) initial = some left ∧
          runSchedule (candidate.schedule vals) initial = some right ∧
          io.OutputsEqual candidate vals left right ∧
          io.Frame vals initial left ∧ io.Frame vals initial right) :
    io.EquivalentOn candidate domain := by
  intro vals hd bounds hb initial hin
  exact ⟨leftSafe vals hd bounds hb initial hin,
    rightSafe vals hd bounds hb initial hin, runs vals hd bounds hb initial hin⟩

/-- Two complete implementations of the same independent logical function
supply an equivalence proof without adding any numerical assumption. -/
theorem EquivalentOn.of_implementsOn {α β : Type} [Inhabited β]
    {io : ProgramIO α β} {candidate : Variant io}
    {domain : io.Inputs → Prop} {f : io.Inputs → io.Outputs}
    (left : io.ImplementsOn domain f)
    (right : candidate.toProgramIO.ImplementsOn domain f) :
    io.EquivalentOn candidate domain := by
  intro vals hd bounds hb initial hin
  obtain ⟨leftSafe, l, leftRun, leftOut, leftFrame⟩ := left vals hd bounds hb initial hin
  obtain ⟨rightSafe, r, rightRun, rightOut, rightFrame⟩ := right vals hd bounds hb initial hin
  refine ⟨leftSafe, rightSafe, l, r, leftRun, rightRun, ?_, leftFrame, rightFrame⟩
  intro o j active
  exact (leftOut o j active).trans (rightOut o j active).symm

end
end ProgramIO
end VeriTile.Triton.FP.Structural
