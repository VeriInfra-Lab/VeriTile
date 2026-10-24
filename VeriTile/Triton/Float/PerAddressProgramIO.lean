import VeriTile.Triton.Float.MemoryIOJudgments
import VeriTile.Triton.Float.InputBindingIO

/-! Per-address observation of complete Structural source executions.

This bounded interface retains independent event orders at different output
addresses. It is not a generic concurrent GPU semantics. Each event executes
its complete concrete Invocation, with the current selected typed cell in the
actual source input state. The next cell is exactly that successful source
result. Source-output locality is a required proved conclusion: changing other
output values must not change the selected result. Every listed CTA must also
execute safely and successfully, including when there are no output lanes.

Task adapters must connect these source-derived transitions to their existing
local atomic linearization boundary. No runner, patch or safety callback is
part of the interface. Private inter-event scratch is deliberately unsupported.
-/
namespace VeriTile.Triton.FP.Structural
noncomputable section

namespace MemoryIO

/-- A dependent name for one declared output lane. -/
abbrev OutputKey {α : Type} (io : MemoryIO α) :=
  (o : Fin io.outputs.length) × TileIndex (io.outputs[o]).shape

abbrev RawOutputs {α : Type} (io : MemoryIO α) (β : Type) :=
  (key : io.OutputKey) → Value β (io.outputs[key.1]).dtype

def outputAddress {α : Type} (io : MemoryIO α) (vals : io.Inputs)
    (key : io.OutputKey) : RegionName × Nat :=
  (io.outputs[key.1].buf, io.write key.1 vals key.2)

def outputActive {α : Type} (io : MemoryIO α) (vals : io.Inputs)
    (key : io.OutputKey) : Prop := io.outputMask key.1 vals key.2

def OutputUnique {α : Type} (io : MemoryIO α) (vals : io.Inputs) : Prop :=
  ∀ left right, io.outputActive vals left → io.outputActive vals right →
    io.outputAddress vals left = io.outputAddress vals right → left = right

/-- Exact tags and encoded values; no constraint on inactive input cells. -/
def InputCells {α β : Type} (io : MemoryIO α) (encode : α → β)
    (vals : io.Inputs) (initial : State β) : Prop :=
  ∀ i j, io.inputMask i vals j →
    initial.mem io.inputs[i].buf (io.read i vals j) =
      Cell.mk io.inputs[i].dtype (mapValue encode io.inputs[i].dtype (vals i j))

/-- The current values of typed output cells, including in-place inputs. -/
def OutputCells {α β : Type} (io : MemoryIO α) (vals : io.Inputs)
    (values : io.RawOutputs β) (s : State β) : Prop :=
  ∀ key, io.outputActive vals key →
    s.mem (io.outputAddress vals key).1 (io.outputAddress vals key).2 =
      Cell.mk (io.outputs[key.1]).dtype (values key)

def OutputHit {α : Type} (io : MemoryIO α) (vals : io.Inputs)
    (region : RegionName) (offset : Nat) : Prop :=
  ∃ key, io.outputActive vals key ∧ io.outputAddress vals key = (region, offset)

/-- Fixed typed output patch; all other memory and all local state are kept. -/
def patchOutputs {α β : Type} (io : MemoryIO α) (vals : io.Inputs)
    (values : io.RawOutputs β) (s : State β) : State β := by
  classical
  exact { s with mem := fun r off =>
    if hit : io.OutputHit vals r off then
      (Cell.mk (io.outputs[(Classical.choose hit).1]).dtype (values (Classical.choose hit)))
    else (s.mem r off) }

theorem patchOutputs_read {α β : Type} (io : MemoryIO α) (vals : io.Inputs)
    (unique : io.OutputUnique vals) (values : io.RawOutputs β) (s : State β)
    (key : io.OutputKey) (active : io.outputActive vals key) :
    (io.patchOutputs vals values s).mem (io.outputAddress vals key).1
      (io.outputAddress vals key).2 = Cell.mk (io.outputs[key.1]).dtype (values key) := by
  classical
  have hit : io.OutputHit vals (io.outputAddress vals key).1 (io.outputAddress vals key).2 :=
    ⟨key, active, rfl⟩
  simp only [patchOutputs, dif_pos hit]
  have selected := Classical.choose_spec hit
  have same : Classical.choose hit = key := unique _ _ selected.1 active selected.2
  exact congrArg (fun k : io.OutputKey => Cell.mk (io.outputs[k.1]).dtype (values k)) same

theorem patchOutputs_cells {α β : Type} (io : MemoryIO α) (vals : io.Inputs)
    (unique : io.OutputUnique vals) (values : io.RawOutputs β) (s : State β) :
    io.OutputCells vals values (io.patchOutputs vals values s) :=
  fun key active => io.patchOutputs_read vals unique values s key active

theorem patchOutputs_frame {α β : Type} (io : MemoryIO α) (vals : io.Inputs)
    (values : io.RawOutputs β) (s : State β) :
    io.Frame vals s (io.patchOutputs vals values s) := by
  classical
  intro region offset outside
  have miss : ¬ io.OutputHit vals region offset := by
    rintro ⟨key, active, address⟩
    have other := outside.1 key.1 key.2 active
    have hr := congrArg Prod.fst address
    have ho := congrArg Prod.snd address
    exact other.elim (fun ne => ne hr.symm) (fun ne => ne ho.symm)
  simp only [patchOutputs, dif_neg miss]

theorem patchOutputs_same {α β : Type} (io : MemoryIO α) (vals : io.Inputs)
    (values : io.RawOutputs β) (s : State β) (bound : io.OutputCells vals values s) :
    io.patchOutputs vals values s = s := by
  classical
  have memory : (io.patchOutputs vals values s).mem = s.mem := by
    funext region offset
    by_cases hit : io.OutputHit vals region offset
    · simp only [patchOutputs, dif_pos hit]
      have selected := Classical.choose_spec hit
      have value := bound (Classical.choose hit) selected.1
      rw [selected.2] at value
      exact value.symm
    · simp only [patchOutputs, dif_neg hit]
  cases s
  simp_all only [patchOutputs]

def updateOutput {α β : Type} (io : MemoryIO α) (values : io.RawOutputs β)
    (key : io.OutputKey) (value : Value β (io.outputs[key.1]).dtype) : io.RawOutputs β := by
  classical
  exact Function.update values key value

@[simp] theorem updateOutput_same {α β : Type} (io : MemoryIO α) (values : io.RawOutputs β)
    (key : io.OutputKey) (value : Value β (io.outputs[key.1]).dtype) :
    io.updateOutput values key value key = value := by
  classical
  exact Function.update_self key value values

end MemoryIO

theorem ProgramIO.inputCells_memory_iff {α β : Type} (io : ProgramIO α β)
    (vals : io.Inputs) (s : State β) :
    io.InputCells vals s ↔ io.toMemoryIO.InputCells io.encode vals s := Iff.rfl

/-- Source events and both canonical and selected per-address orders are data.
The independent output carrier prevents a native observation from restricting
the logical input carrier to literal leaves. -/
structure PerAddressProgramIO (α β γ Event : Type) extends MemoryIO α where
  programs : ChannelValues α inputs → List Event
  invocation : ChannelValues α inputs → Event → Invocation β
  events : ChannelValues α inputs → (o : Fin outputs.length) →
    TileIndex (outputs[o]).shape → List Event
  order : ChannelValues α inputs → (o : Fin outputs.length) →
    TileIndex (outputs[o]).shape → List Event
  encode : α → β
  observe : (o : Fin outputs.length) → TileIndex (outputs[o]).shape → β → γ

namespace PerAddressProgramIO
variable {α β γ Event : Type}

abbrev Inputs (io : PerAddressProgramIO α β γ Event) := io.toMemoryIO.Inputs
abbrev Outputs (io : PerAddressProgramIO α β γ Event) := ChannelValues γ io.outputs
abbrev OutputKey (io : PerAddressProgramIO α β γ Event) := io.toMemoryIO.OutputKey
abbrev RawOutputs (io : PerAddressProgramIO α β γ Event) := io.toMemoryIO.RawOutputs β
abbrev InBounds (io : PerAddressProgramIO α β γ Event) := io.toMemoryIO.InBounds
abbrev InputCells (io : PerAddressProgramIO α β γ Event) := io.toMemoryIO.InputCells io.encode
abbrev OutputCells (io : PerAddressProgramIO α β γ Event) := @MemoryIO.OutputCells α β io.toMemoryIO
abbrev Frame (io : PerAddressProgramIO α β γ Event) := @MemoryIO.Frame α β io.toMemoryIO
abbrev patchOutputs (io : PerAddressProgramIO α β γ Event) := @MemoryIO.patchOutputs α β io.toMemoryIO
abbrev outputAddress (io : PerAddressProgramIO α β γ Event) := io.toMemoryIO.outputAddress
abbrev outputActive (io : PerAddressProgramIO α β γ Event) := io.toMemoryIO.outputActive

def OutputBinding (io : PerAddressProgramIO α β γ Event) (vals : io.Inputs)
    (expected : io.Outputs) (s : State β) : Prop :=
  ∀ o j, io.outputMask o vals j →
    mapCell (io.observe o j) (s.mem io.outputs[o].buf (io.write o vals j)) =
      Cell.mk io.outputs[o].dtype (expected o j)

def ScheduleComplete (io : PerAddressProgramIO α β γ Event) (vals : io.Inputs) : Prop := by
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
def Programs [Inhabited β] (io : PerAddressProgramIO α β γ Event)
    (vals : io.Inputs) (base : State β) : Prop :=
  ∀ current : io.RawOutputs, ∀ event ∈ io.programs vals,
    (∀ bounds, io.InBounds bounds vals →
      (io.invocation vals event).Safe bounds (io.patchOutputs vals current base)) ∧
    ∃ final, (io.invocation vals event).run (io.patchOutputs vals current base) = some final ∧
      io.Frame vals (io.patchOutputs vals current base) final

/-- A proved source property, not a numerical or runtime premise. Other
declared output values may vary arbitrarily while the selected value is kept.
Success for every compared run is separately guaranteed by Programs. -/
def SourceLocality [Inhabited β] (io : PerAddressProgramIO α β γ Event)
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
def SourceCoverage [Inhabited β] (io : PerAddressProgramIO α β γ Event)
    (vals : io.Inputs) (base : State β) : Prop :=
  ∀ key : io.OutputKey, io.outputActive vals key →
  ∀ event ∈ io.programs vals, event ∉ io.events vals key.1 key.2 →
  ∀ current : io.RawOutputs, ∀ final,
    (io.invocation vals event).run (io.patchOutputs vals current base) = some final →
    final.mem (io.outputAddress vals key).1 (io.outputAddress vals key).2 =
      Cell.mk (io.outputs[key.1]).dtype (current key)

def currentState (io : PerAddressProgramIO α β γ Event) (vals : io.Inputs)
    (base : State β) (initial : io.RawOutputs) (key : io.OutputKey)
    (current : Value β (io.outputs[key.1]).dtype) : State β :=
  io.patchOutputs vals (io.toMemoryIO.updateOutput initial key current) base

/-- Each transition uses the exact selected Cell of a successful full source
execution. No default-valued read can conceal a wrong output dtype. -/
inductive SourceCellTrace [Inhabited β] (io : PerAddressProgramIO α β γ Event)
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

theorem SourceCellTrace.nil_iff [Inhabited β] (io : PerAddressProgramIO α β γ Event)
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
    (io : PerAddressProgramIO α β γ Event) (vals : io.Inputs) (base : State β)
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
def ImplementsOnCells [Inhabited β] (io : PerAddressProgramIO α β γ Event)
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

def ImplementsCells [Inhabited β] (io : PerAddressProgramIO α β γ Event)
    (f : io.Inputs → io.Outputs) : Prop := io.ImplementsOnCells (fun _ => True) f

end PerAddressProgramIO
end
end VeriTile.Triton.FP.Structural
