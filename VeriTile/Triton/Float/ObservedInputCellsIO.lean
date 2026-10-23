import VeriTile.Triton.Float.ObservedProgramIO
import VeriTile.Triton.Float.InputBindingIO

/-! Pure output observation with the existing exact-cell input convention.
This is the exact-cell counterpart of `ObservesOn`: inputs, source execution,
access safety, output dtype and the complete raw exterior frame are unchanged.
The additional observer sees only one logical floating payload at a time.
-/
namespace VeriTile.Triton.FP.Structural.ProgramIO
noncomputable section

def ObservesOnCells {α β γ : Type} [Inhabited β] (io : ProgramIO α β)
    (observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → α → γ)
    (domain : io.Inputs → Prop) (f : io.Inputs → io.ObservedOutputs γ) : Prop :=
  ∀ vals, domain vals → ∀ bounds, io.InBounds bounds vals → ∀ initial,
    io.InputCells vals initial →
    ScheduleSafe bounds (io.schedule vals) initial ∧
    ∃ final, runSchedule (io.schedule vals) initial = some final ∧
      io.ObservedOutputBinding observe vals (f vals) final ∧ io.Frame vals initial final

def ObservesCells {α β γ : Type} [Inhabited β] (io : ProgramIO α β)
    (observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → α → γ)
    (f : io.Inputs → io.ObservedOutputs γ) : Prop :=
  io.ObservesOnCells observe (fun _ => True) f

/-- Compose a complete exact-cell implementation with a pure observation.
No execution, safety or exterior-memory condition is removed. -/
theorem ObservesOnCells.of_implementsOnCells {α β γ : Type} [Inhabited β]
    {io : ProgramIO α β} {domain : io.Inputs → Prop} {f : io.Inputs → io.Outputs}
    (implemented : io.ImplementsOnCells domain f)
    (observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → α → γ) :
    io.ObservesOnCells observe domain (fun vals => io.mapOutputs observe (f vals)) := by
  intro vals hd bounds hb initial hin
  obtain ⟨safe, final, run, output, frame⟩ := implemented vals hd bounds hb initial hin
  exact ⟨safe, final, run, output.observe observe, frame⟩

theorem ObservesCells.of_implementsCells {α β γ : Type} [Inhabited β]
    {io : ProgramIO α β} {f : io.Inputs → io.Outputs}
    (implemented : io.ImplementsCells f)
    (observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → α → γ) :
    io.ObservesCells observe (fun vals => io.mapOutputs observe (f vals)) :=
  ObservesOnCells.of_implementsOnCells implemented observe

end
end VeriTile.Triton.FP.Structural.ProgramIO
