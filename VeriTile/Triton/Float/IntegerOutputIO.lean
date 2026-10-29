import VeriTile.Triton.Float.IntegerStageIO

/-! Closed integer output views for the existing integer-stage source executor.

Some source storage adapters retain a wider integer cell while the declared
external observation is a fixed-width byte/word. These views alter only the
observed `.int` payload; dtype tags and every other channel remain observable.
The executor, source safety, input domain and raw exterior frame are unchanged.
No arbitrary integer function, execution callback or safety premise is stored.
-/
namespace VeriTile.Triton.FP.Structural
noncomputable section

inductive IntegerOutputView where
  | identity
  | signedWidth (bits : Nat)
  | unsignedWidth (bits : Nat)
  deriving DecidableEq, Repr

namespace IntegerOutputView

def apply : IntegerOutputView → Int → Int
  | .identity, value => value
  | .signedWidth bits, value => (BitVec.ofInt bits value).toInt
  | .unsignedWidth bits, value => Int.ofNat (BitVec.ofInt bits value).toNat

def mapValue {α : Type} (view : IntegerOutputView) :
    (dtype : TileDType) → Value α dtype → Value α dtype
  | .int => view.apply
  | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 | .nat | .bool | .ptr | .blockPtr => id

def mapCell {α : Type} (view : IntegerOutputView) : Cell α → Cell α
  | .mk dtype value => .mk dtype (view.mapValue dtype value)

end IntegerOutputView

namespace MemoryIO

def mapIntegerOutputs {α : Type} (io : MemoryIO α)
    (views : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → IntegerOutputView)
    (outputs : io.Outputs) : io.Outputs :=
  fun o j => (views o j).mapValue (io.outputs[o]).dtype (outputs o j)

def IntegerOutputBinding {α β : Type} (io : MemoryIO α)
    (observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → β → α)
    (views : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → IntegerOutputView)
    (vals : io.Inputs) (expected : io.Outputs) (s : State β) : Prop :=
  ∀ o j, io.outputMask o vals j →
    (views o j).mapCell
      (mapCell (observe o j) (s.mem io.outputs[o].buf (io.write o vals j))) =
      Cell.mk io.outputs[o].dtype (expected o j)

theorem OutputBinding.observeInteger {α β : Type} {io : MemoryIO α}
    {observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → β → α}
    {vals : io.Inputs} {expected : io.Outputs} {s : State β}
    (bound : io.OutputBinding observe vals expected s)
    (views : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → IntegerOutputView) :
    io.IntegerOutputBinding observe views vals (io.mapIntegerOutputs views expected) s := by
  intro o j active
  exact congrArg (IntegerOutputView.mapCell (views o j)) (bound o j active)

end MemoryIO

namespace IntegerProgramIO

abbrev IntegerOutputBinding {α β : Type} (io : IntegerProgramIO α β)
    (views : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → IntegerOutputView) :=
  io.toMemoryIO.IntegerOutputBinding io.observe views

def ObservesIntegerOn {α β : Type} [Inhabited β] (io : IntegerProgramIO α β)
    (views : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → IntegerOutputView)
    (domain : io.Inputs → Prop) (f : io.Inputs → io.Outputs) : Prop :=
  ∀ vals, domain vals → ∀ bounds, io.InBounds bounds vals → ∀ initial,
    io.InputBinding vals initial →
    IntegerScheduleSafe bounds (io.schedule vals) initial ∧
    ∃ final, runIntegerSchedule (io.schedule vals) initial = some final ∧
      io.IntegerOutputBinding views vals (f vals) final ∧ io.Frame vals initial final

def ObservesInteger {α β : Type} [Inhabited β] (io : IntegerProgramIO α β)
    (views : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → IntegerOutputView)
    (f : io.Inputs → io.Outputs) : Prop :=
  io.ObservesIntegerOn views (fun _ => True) f

theorem ObservesIntegerOn.of_implementsOn {α β : Type} [Inhabited β]
    {io : IntegerProgramIO α β} {domain : io.Inputs → Prop} {f : io.Inputs → io.Outputs}
    (implemented : io.ImplementsOn domain f)
    (views : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → IntegerOutputView) :
    io.ObservesIntegerOn views domain (fun vals => io.toMemoryIO.mapIntegerOutputs views (f vals)) := by
  intro vals admitted bounds backed initial inputs
  obtain ⟨safe, final, ran, outputs, frame⟩ :=
    implemented vals admitted bounds backed initial inputs
  exact ⟨safe, final, ran, outputs.observeInteger views, frame⟩

theorem ObservesInteger.of_implements {α β : Type} [Inhabited β]
    {io : IntegerProgramIO α β} {f : io.Inputs → io.Outputs}
    (implemented : io.Implements f)
    (views : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → IntegerOutputView) :
    io.ObservesInteger views (fun vals => io.toMemoryIO.mapIntegerOutputs views (f vals)) :=
  ObservesIntegerOn.of_implementsOn implemented views

theorem ObservesInteger.intro {α β : Type} [Inhabited β] {io : IntegerProgramIO α β}
    {views : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → IntegerOutputView}
    {f : io.Inputs → io.Outputs}
    (safe : ∀ vals bounds, io.InBounds bounds vals → ∀ initial,
      io.InputBinding vals initial → IntegerScheduleSafe bounds (io.schedule vals) initial)
    (correct : ∀ vals initial, io.InputBinding vals initial →
      ∃ final, runIntegerSchedule (io.schedule vals) initial = some final ∧
        io.IntegerOutputBinding views vals (f vals) final ∧ io.Frame vals initial final) :
    io.ObservesInteger views f := by
  intro vals _ bounds backed initial inputs
  exact ⟨safe vals bounds backed initial inputs, correct vals initial inputs⟩

def IntegerOutputsEqual {α β : Type} (io : IntegerProgramIO α β) (candidate : Variant io)
    (views : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → IntegerOutputView)
    (vals : io.Inputs) (left right : State β) : Prop :=
  ∀ o j, io.outputMask o vals j →
    (views o j).mapCell
      (mapCell (io.observe o j) (left.mem io.outputs[o].buf (io.write o vals j))) =
    (views o j).mapCell
      (mapCell (candidate.observe o j) (right.mem io.outputs[o].buf (io.write o vals j)))

def IntegerObservedEquivalentOn {α β : Type} [Inhabited β] (io : IntegerProgramIO α β)
    (candidate : Variant io)
    (views : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → IntegerOutputView)
    (domain : io.Inputs → Prop) : Prop :=
  ∀ vals, domain vals → ∀ bounds, io.InBounds bounds vals → ∀ initial,
    io.InputBinding vals initial →
    IntegerScheduleSafe bounds (io.schedule vals) initial ∧
    IntegerScheduleSafe bounds (candidate.schedule vals) initial ∧
    ∃ left right,
      runIntegerSchedule (io.schedule vals) initial = some left ∧
      runIntegerSchedule (candidate.schedule vals) initial = some right ∧
      io.IntegerOutputsEqual candidate views vals left right ∧
      io.Frame vals initial left ∧ io.Frame vals initial right

def IntegerObservedEquivalent {α β : Type} [Inhabited β] (io : IntegerProgramIO α β)
    (candidate : Variant io)
    (views : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → IntegerOutputView) : Prop :=
  io.IntegerObservedEquivalentOn candidate views (fun _ => True)

theorem IntegerObservedEquivalentOn.of_equivalentOn {α β : Type} [Inhabited β]
    {io : IntegerProgramIO α β} {candidate : Variant io} {domain : io.Inputs → Prop}
    (equivalent : io.EquivalentOn candidate domain)
    (views : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → IntegerOutputView) :
    io.IntegerObservedEquivalentOn candidate views domain := by
  intro vals admitted bounds backed initial inputs
  obtain ⟨lsafe, rsafe, left, right, lrun, rrun, equal, lframe, rframe⟩ :=
    equivalent vals admitted bounds backed initial inputs
  refine ⟨lsafe, rsafe, left, right, lrun, rrun, ?_, lframe, rframe⟩
  intro o j active
  exact congrArg (IntegerOutputView.mapCell (views o j)) (equal o j active)

theorem IntegerObservedEquivalentOn.of_observesIntegerOn {α β : Type} [Inhabited β]
    {io : IntegerProgramIO α β} {candidate : Variant io} {domain : io.Inputs → Prop}
    {views : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → IntegerOutputView}
    {f : io.Inputs → io.Outputs}
    (left : io.ObservesIntegerOn views domain f)
    (right : candidate.toIntegerProgramIO.ObservesIntegerOn views domain f) :
    io.IntegerObservedEquivalentOn candidate views domain := by
  intro vals admitted bounds backed initial inputs
  obtain ⟨lsafe, left, lrun, lout, lframe⟩ := left vals admitted bounds backed initial inputs
  obtain ⟨rsafe, right, rrun, rout, rframe⟩ := right vals admitted bounds backed initial inputs
  refine ⟨lsafe, rsafe, left, right, lrun, rrun, ?_, lframe, rframe⟩
  intro o j active
  exact (lout o j active).trans (rout o j active).symm

end IntegerProgramIO
end
end VeriTile.Triton.FP.Structural
