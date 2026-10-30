import VeriTile.Triton.Float.ProgramIO

/-! A separate output observation for an existing source-connected IO.

Logical inputs and actual execution keep their original carriers. Only the
declared output payloads are mapped to another carrier, independently for each
output and lane. This supports native evaluation of symbolic output terms
without restricting symbolic inputs to leaves or changing their binding.
The observer is a pure payload function: it cannot inspect or replace a state,
source execution, safety obligation, or exterior frame. Discrete payloads and
cell dtype tags are preserved by the existing `mapValue`/`mapCell` operations.
-/
namespace VeriTile.Triton.FP.Structural.ProgramIO
noncomputable section

abbrev ObservedOutputs {α β : Type} (io : ProgramIO α β) (γ : Type) :=
  ChannelValues γ io.outputs

/-- Apply an output/lane-specific observation to floating logical values only. -/
def mapOutputs {α β γ : Type} (io : ProgramIO α β)
    (observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → α → γ)
    (expected : io.Outputs) : io.ObservedOutputs γ :=
  fun o j => mapValue (observe o j) io.outputs[o].dtype (expected o j)

/-- The same typed active output cells, with an additional pure observation.
Exact dtype agreement and all discrete payloads remain observable. -/
def ObservedOutputBinding {α β γ : Type} (io : ProgramIO α β)
    (observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → α → γ)
    (vals : io.Inputs) (expected : io.ObservedOutputs γ) (s : State β) : Prop :=
  ∀ o j, io.outputMask o vals j →
    mapCell (observe o j)
      (mapCell (io.observe o j) (s.mem io.outputs[o].buf (io.write o vals j))) =
        Cell.mk io.outputs[o].dtype (expected o j)

/-- Actual source safety, successful execution, observed outputs, and the
complete original frame, under the original logical input domain/binding. -/
def ObservesOn {α β γ : Type} [Inhabited β] (io : ProgramIO α β)
    (observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → α → γ)
    (domain : io.Inputs → Prop) (f : io.Inputs → io.ObservedOutputs γ) : Prop :=
  ∀ vals, domain vals → ∀ bounds, io.InBounds bounds vals → ∀ initial,
    io.InputBinding vals initial →
    ScheduleSafe bounds (io.schedule vals) initial ∧
    ∃ final, runSchedule (io.schedule vals) initial = some final ∧
      io.ObservedOutputBinding observe vals (f vals) final ∧ io.Frame vals initial final

def Observes {α β γ : Type} [Inhabited β] (io : ProgramIO α β)
    (observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → α → γ)
    (f : io.Inputs → io.ObservedOutputs γ) : Prop :=
  io.ObservesOn observe (fun _ => True) f

theorem OutputBinding.observe {α β γ : Type} {io : ProgramIO α β}
    {vals : io.Inputs} {expected : io.Outputs} {s : State β}
    (bound : io.OutputBinding vals expected s)
    (observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → α → γ) :
    io.ObservedOutputBinding observe vals (io.mapOutputs observe expected) s := by
  intro o j active
  exact congrArg (mapCell (observe o j)) (bound o j active)

/-- Compose a complete implementation theorem with a pure output observation.
No change to the input domain, encoding, source schedule, or frame is needed. -/
theorem ObservesOn.of_implementsOn {α β γ : Type} [Inhabited β]
    {io : ProgramIO α β} {domain : io.Inputs → Prop} {f : io.Inputs → io.Outputs}
    (implemented : io.ImplementsOn domain f)
    (observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → α → γ) :
    io.ObservesOn observe domain (fun vals => io.mapOutputs observe (f vals)) := by
  intro vals hd bounds hb initial hin
  obtain ⟨safe, final, run, output, frame⟩ := implemented vals hd bounds hb initial hin
  exact ⟨safe, final, run, output.observe observe, frame⟩

theorem Observes.of_implements {α β γ : Type} [Inhabited β]
    {io : ProgramIO α β} {f : io.Inputs → io.Outputs}
    (implemented : io.Implements f)
    (observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → α → γ) :
    io.Observes observe (fun vals => io.mapOutputs observe (f vals)) :=
  ObservesOn.of_implementsOn implemented observe

end
end VeriTile.Triton.FP.Structural.ProgramIO
