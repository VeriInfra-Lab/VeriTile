import VeriTile.Triton.Float.ProgramEquivalence
import VeriTile.Triton.Float.ObservedProgramIO

/-! Separate native output observation for two existing source schedules.
Input carriers, domains and encodings stay unchanged. Both schedules must
execute successfully with actual source safety and exact exterior frames.
Only complete active output cells undergo a pure output/lane observation;
dtype tags and discrete payloads remain observable.
-/
namespace VeriTile.Triton.FP.Structural.ProgramIO
noncomputable section

def ObservedOutputsEqual {α β γ : Type} (io : ProgramIO α β) (candidate : Variant io)
    (observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → α → γ)
    (vals : io.Inputs) (left right : State β) : Prop :=
  ∀ o j, io.outputMask o vals j →
    mapCell (observe o j)
      (mapCell (io.observe o j) (left.mem io.outputs[o].buf (io.write o vals j))) =
    mapCell (observe o j)
      (mapCell (candidate.observe o j) (right.mem io.outputs[o].buf (io.write o vals j)))

def ObservedEquivalentOn {α β γ : Type} [Inhabited β] (io : ProgramIO α β)
    (candidate : Variant io)
    (observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → α → γ)
    (domain : io.Inputs → Prop) : Prop :=
  ∀ vals, domain vals → ∀ bounds, io.InBounds bounds vals → ∀ initial,
    io.InputBinding vals initial →
    ScheduleSafe bounds (io.schedule vals) initial ∧
    ScheduleSafe bounds (candidate.schedule vals) initial ∧
    ∃ left right,
      runSchedule (io.schedule vals) initial = some left ∧
      runSchedule (candidate.schedule vals) initial = some right ∧
      io.ObservedOutputsEqual candidate observe vals left right ∧
      io.Frame vals initial left ∧ io.Frame vals initial right

def ObservedEquivalent {α β γ : Type} [Inhabited β] (io : ProgramIO α β)
    (candidate : Variant io)
    (observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → α → γ) : Prop :=
  io.ObservedEquivalentOn candidate observe (fun _ => True)

theorem ObservedEquivalentOn.of_equivalentOn {α β γ : Type} [Inhabited β]
    {io : ProgramIO α β} {candidate : Variant io} {domain : io.Inputs → Prop}
    (equivalent : io.EquivalentOn candidate domain)
    (observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → α → γ) :
    io.ObservedEquivalentOn candidate observe domain := by
  intro vals hd bounds hb initial hin
  obtain ⟨lsafe, rsafe, l, r, lrun, rrun, equal, lframe, rframe⟩ :=
    equivalent vals hd bounds hb initial hin
  refine ⟨lsafe, rsafe, l, r, lrun, rrun, ?_, lframe, rframe⟩
  intro o j active
  exact congrArg (mapCell (observe o j)) (equal o j active)

theorem ObservedEquivalent.of_equivalent {α β γ : Type} [Inhabited β]
    {io : ProgramIO α β} {candidate : Variant io}
    (equivalent : io.Equivalent candidate)
    (observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → α → γ) :
    io.ObservedEquivalent candidate observe :=
  ObservedEquivalentOn.of_equivalentOn equivalent observe

/-- Two complete implementations of the same observed target can have
different symbolic outputs; both original source runs and frames are retained. -/
theorem ObservedEquivalentOn.of_observesOn {α β γ : Type} [Inhabited β]
    {io : ProgramIO α β} {candidate : Variant io} {domain : io.Inputs → Prop}
    {observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → α → γ}
    {f : io.Inputs → io.ObservedOutputs γ}
    (left : io.ObservesOn observe domain f)
    (right : candidate.toProgramIO.ObservesOn observe domain f) :
    io.ObservedEquivalentOn candidate observe domain := by
  intro vals hd bounds hb initial hin
  obtain ⟨lsafe, l, lrun, lout, lframe⟩ := left vals hd bounds hb initial hin
  obtain ⟨rsafe, r, rrun, rout, rframe⟩ := right vals hd bounds hb initial hin
  refine ⟨lsafe, rsafe, l, r, lrun, rrun, ?_, lframe, rframe⟩
  intro o j active
  exact (lout o j active).trans (rout o j active).symm

theorem ObservedEquivalent.of_observes {α β γ : Type} [Inhabited β]
    {io : ProgramIO α β} {candidate : Variant io}
    {observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → α → γ}
    {f : io.Inputs → io.ObservedOutputs γ}
    (left : io.Observes observe f)
    (right : candidate.toProgramIO.Observes observe f) :
    io.ObservedEquivalent candidate observe :=
  ObservedEquivalentOn.of_observesOn left right

end
end VeriTile.Triton.FP.Structural.ProgramIO
