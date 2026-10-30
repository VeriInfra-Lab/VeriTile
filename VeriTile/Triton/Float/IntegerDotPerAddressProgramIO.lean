import VeriTile.Triton.Float.PerAddressProgramIO
import VeriTile.Triton.Float.IntegerDotIO

/-! Per-address observations of complete contextual integer-dot source launches.

The concrete executor and scalar topology are fixed by `IntegerDotInvocation`.
This is the integer-dot specialization of `PerAddressProgramIO`: complete CTA
success and access safety, selected-cell locality, coverage of every updating
source event, exact source-derived cell traces, and the exterior frame remain
mandatory conclusions. No execution, state-transform, or safety callback is
accepted. The original per-address atomic linearization remains an explicit
task-level transport obligation.
-/
namespace VeriTile.Triton.FP.Structural
noncomputable section

/-- Source events and both canonical and selected per-address orders are data.
The independent output carrier prevents a native observation from restricting
the logical input carrier to literal leaves. -/
structure IntegerDotPerAddressProgramIO (α β γ Event : Type) extends MemoryIO α where
  programs : ChannelValues α inputs → List Event
  invocation : ChannelValues α inputs → Event → IntegerDotInvocation β
  events : ChannelValues α inputs → (o : Fin outputs.length) →
    TileIndex (outputs[o]).shape → List Event
  order : ChannelValues α inputs → (o : Fin outputs.length) →
    TileIndex (outputs[o]).shape → List Event
  encode : α → β
  observe : (o : Fin outputs.length) → TileIndex (outputs[o]).shape → β → γ

namespace IntegerDotPerAddressProgramIO
variable {α β γ Event : Type}

abbrev Inputs (io : IntegerDotPerAddressProgramIO α β γ Event) := io.toMemoryIO.Inputs
abbrev Outputs (io : IntegerDotPerAddressProgramIO α β γ Event) := ChannelValues γ io.outputs
abbrev OutputKey (io : IntegerDotPerAddressProgramIO α β γ Event) := io.toMemoryIO.OutputKey
abbrev RawOutputs (io : IntegerDotPerAddressProgramIO α β γ Event) := io.toMemoryIO.RawOutputs β
abbrev InBounds (io : IntegerDotPerAddressProgramIO α β γ Event) := io.toMemoryIO.InBounds
abbrev InputCells (io : IntegerDotPerAddressProgramIO α β γ Event) := io.toMemoryIO.InputCells io.encode
abbrev OutputCells (io : IntegerDotPerAddressProgramIO α β γ Event) := @MemoryIO.OutputCells α β io.toMemoryIO
abbrev Frame (io : IntegerDotPerAddressProgramIO α β γ Event) := @MemoryIO.Frame α β io.toMemoryIO
abbrev patchOutputs (io : IntegerDotPerAddressProgramIO α β γ Event) := @MemoryIO.patchOutputs α β io.toMemoryIO
abbrev outputAddress (io : IntegerDotPerAddressProgramIO α β γ Event) := io.toMemoryIO.outputAddress
abbrev outputActive (io : IntegerDotPerAddressProgramIO α β γ Event) := io.toMemoryIO.outputActive

def OutputBinding (io : IntegerDotPerAddressProgramIO α β γ Event) (vals : io.Inputs)
    (expected : io.Outputs) (s : State β) : Prop :=
  ∀ o j, io.outputMask o vals j →
    mapCell (io.observe o j) (s.mem io.outputs[o].buf (io.write o vals j)) =
      Cell.mk io.outputs[o].dtype (expected o j)

def ScheduleComplete (io : IntegerDotPerAddressProgramIO α β γ Event) (vals : io.Inputs) : Prop := by
  classical
  exact
  io.scratch = [] ∧ io.toMemoryIO.OutputUnique vals ∧
  ∀ key : io.OutputKey, io.outputActive vals key →
    (io.order vals key.1 key.2).Perm (io.events vals key.1 key.2) ∧
    (io.events vals key.1 key.2 ++
      (io.programs vals).filter (fun event => decide (event ∉ io.events vals key.1 key.2))).Perm
      (io.programs vals)

/-- Check every complete CTA at every typed current output, even if outputs
are empty. Safety is conditional only on capacities; source success is not. -/
def Programs [Inhabited β] (io : IntegerDotPerAddressProgramIO α β γ Event)
    (vals : io.Inputs) (base : State β) : Prop :=
  ∀ current : io.RawOutputs, ∀ event ∈ io.programs vals,
    (∀ bounds, io.InBounds bounds vals →
      (io.invocation vals event).Safe bounds (io.patchOutputs vals current base)) ∧
    ∃ final, (io.invocation vals event).run (io.patchOutputs vals current base) = some final ∧
      io.Frame vals (io.patchOutputs vals current base) final

/-- A proved source property, not a numerical or runtime premise. Other
declared output values may vary arbitrarily while the selected value is kept.
Success for every compared run is separately guaranteed by Programs. -/
def SourceLocality [Inhabited β] (io : IntegerDotPerAddressProgramIO α β γ Event)
    (vals : io.Inputs) (base : State β) : Prop :=
  ∀ event ∈ io.programs vals, ∀ key : io.OutputKey, io.outputActive vals key →
  ∀ left right : io.RawOutputs, left key = right key → ∀ l r,
    (io.invocation vals event).run (io.patchOutputs vals left base) = some l →
    (io.invocation vals event).run (io.patchOutputs vals right base) = some r →
    l.mem (io.outputAddress vals key).1 (io.outputAddress vals key).2 =
      r.mem (io.outputAddress vals key).1 (io.outputAddress vals key).2

/-- An event may be omitted at an address only when its actual complete source
execution leaves that selected cell unchanged for every typed output patch.
Together with the schedule partition, this prevents omitted or duplicated
updating source events from being hidden by a chosen per-address order. -/
def SourceCoverage [Inhabited β] (io : IntegerDotPerAddressProgramIO α β γ Event)
    (vals : io.Inputs) (base : State β) : Prop :=
  ∀ key : io.OutputKey, io.outputActive vals key →
  ∀ event ∈ io.programs vals, event ∉ io.events vals key.1 key.2 →
  ∀ current : io.RawOutputs, ∀ final,
    (io.invocation vals event).run (io.patchOutputs vals current base) = some final →
    final.mem (io.outputAddress vals key).1 (io.outputAddress vals key).2 =
      Cell.mk (io.outputs[key.1]).dtype (current key)

def currentState (io : IntegerDotPerAddressProgramIO α β γ Event) (vals : io.Inputs)
    (base : State β) (initial : io.RawOutputs) (key : io.OutputKey)
    (current : Value β (io.outputs[key.1]).dtype) : State β :=
  io.patchOutputs vals (io.toMemoryIO.updateOutput initial key current) base

/-- Each transition uses the exact selected Cell of a successful full source
execution. No default-valued read can conceal a wrong output dtype. -/
inductive SourceCellTrace [Inhabited β] (io : IntegerDotPerAddressProgramIO α β γ Event)
    (vals : io.Inputs) (base : State β) (initial : io.RawOutputs) (key : io.OutputKey) :
    List Event → Value β (io.outputs[key.1]).dtype → Value β (io.outputs[key.1]).dtype → Prop
  | nil (value) : SourceCellTrace io vals base initial key [] value value
  | cons (event : Event) (rest : List Event) (before middle after) (final : State β)
      (listed : event ∈ io.programs vals)
      (ran : (io.invocation vals event).run (io.currentState vals base initial key before) = some final)
      (cell : final.mem (io.outputAddress vals key).1 (io.outputAddress vals key).2 =
        Cell.mk (io.outputs[key.1]).dtype middle)
      (safe : ∀ bounds, io.InBounds bounds vals →
        (io.invocation vals event).Safe bounds (io.currentState vals base initial key before))
      (frame : io.Frame vals (io.currentState vals base initial key before) final)
      (tail : SourceCellTrace io vals base initial key rest middle after) :
      SourceCellTrace io vals base initial key (event :: rest) before after

theorem SourceCellTrace.nil_iff [Inhabited β] (io : IntegerDotPerAddressProgramIO α β γ Event)
    (vals : io.Inputs) (base : State β) (initial : io.RawOutputs) (key : io.OutputKey)
    (before after : Value β (io.outputs[key.1]).dtype) :
    io.SourceCellTrace vals base initial key [] before after ↔ after = before := by
  constructor
  · intro trace
    cases trace
    rfl
  · intro same
    subst after
    exact .nil before

/-- Reuse the all-source success/safety certificate for one exact source-cell
transition. The selected cell law remains a theorem of the concrete source. -/
theorem SourceCellTrace.cons_of_programs [Inhabited β]
    (io : IntegerDotPerAddressProgramIO α β γ Event) (vals : io.Inputs) (base : State β)
    (initial : io.RawOutputs) (key : io.OutputKey) (event : Event) (rest : List Event)
    (before middle after : Value β (io.outputs[key.1]).dtype)
    (programs : io.Programs vals base) (listed : event ∈ io.programs vals)
    (value : ∀ final,
      (io.invocation vals event).run (io.currentState vals base initial key before) = some final →
      final.mem (io.outputAddress vals key).1 (io.outputAddress vals key).2 =
        Cell.mk (io.outputs[key.1]).dtype middle)
    (tail : io.SourceCellTrace vals base initial key rest middle after) :
    io.SourceCellTrace vals base initial key (event :: rest) before after := by
  obtain ⟨safe, final, ran, frame⟩ :=
    programs (io.toMemoryIO.updateOutput initial key before) event listed
  exact .cons event rest before middle after final listed ran (value final ran) safe frame tail

/-- Independent mathematical/native output target, with the exact old typed
input/output-cell domain. All execution, locality, safety and frame claims are
conclusions. Initial output values are bound to actual memory, not guessed. -/
def ImplementsOnCells [Inhabited β] (io : IntegerDotPerAddressProgramIO α β γ Event)
    (domain : io.Inputs → Prop) (f : io.Inputs → io.Outputs) : Prop :=
  ∀ vals, domain vals → ∀ base, io.InputCells vals base →
  ∀ initial : io.RawOutputs, io.OutputCells vals initial base →
    io.ScheduleComplete vals ∧ io.Programs vals base ∧ io.SourceLocality vals base ∧
    io.SourceCoverage vals base ∧
    ∃ finalValues : io.RawOutputs,
      (∀ key, io.outputActive vals key →
        io.SourceCellTrace vals base initial key (io.order vals key.1 key.2)
          (initial key) (finalValues key)) ∧
      io.OutputBinding vals (f vals) (io.patchOutputs vals finalValues base) ∧
      io.Frame vals base (io.patchOutputs vals finalValues base)

def ImplementsCells [Inhabited β] (io : IntegerDotPerAddressProgramIO α β γ Event)
    (f : io.Inputs → io.Outputs) : Prop := io.ImplementsOnCells (fun _ => True) f

/-- A fixed typed output initialization followed by the same complete-source
per-address semantics. `initial` is finite typed data, never a state transform
or executor. The initializer changes exactly declared active output cells;
input preservation, the initializer frame and all source obligations are
conclusions. This retains reset-before-atomic source contracts without imposing
an initial-output binding on the caller. -/
def ImplementsResetOnCells [Inhabited β] (io : IntegerDotPerAddressProgramIO α β γ Event)
    (domain : io.Inputs → Prop) (initial : io.RawOutputs)
    (f : io.Inputs → io.Outputs) : Prop :=
  ∀ vals, domain vals → ∀ base, io.InputCells vals base →
    let start := io.patchOutputs vals initial base
    io.InputCells vals start ∧ io.OutputCells vals initial start ∧ io.Frame vals base start ∧
    io.ScheduleComplete vals ∧ io.Programs vals start ∧ io.SourceLocality vals start ∧
    io.SourceCoverage vals start ∧
    ∃ finalValues : io.RawOutputs,
      (∀ key, io.outputActive vals key →
        io.SourceCellTrace vals start initial key (io.order vals key.1 key.2)
          (initial key) (finalValues key)) ∧
      io.OutputBinding vals (f vals) (io.patchOutputs vals finalValues start) ∧
      io.Frame vals base (io.patchOutputs vals finalValues start)

def ImplementsResetCells [Inhabited β] (io : IntegerDotPerAddressProgramIO α β γ Event)
    (initial : io.RawOutputs) (f : io.Inputs → io.Outputs) : Prop :=
  io.ImplementsResetOnCells (fun _ => True) initial f

end IntegerDotPerAddressProgramIO
end
end VeriTile.Triton.FP.Structural
