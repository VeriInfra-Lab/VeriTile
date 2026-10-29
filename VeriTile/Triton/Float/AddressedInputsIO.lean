import VeriTile.Triton.Float.ProgramIO

/-! Source IO whose floating input representation records the read address.

Native term interpreters can label an input with its source region and offset.
This interface gives the encoder exactly those two public address components,
along with the logical value. It receives neither a state nor an executor.
Discrete metadata remains unchanged. The actual schedules, source safety,
typed observations and exact exterior frame have the same meanings as ProgramIO.
-/
namespace VeriTile.Triton.FP.Structural
noncomputable section

/-- The shared memory descriptor with addressed floating input encoding.
All execution remains a finite schedule of actual source invocations. -/
structure AddressedProgramIO (α β : Type) extends MemoryIO α where
  schedule : ChannelValues α inputs → List (Invocation β)
  encodeAt : RegionName → Nat → α → β
  observe : (o : Fin outputs.length) → TileIndex (outputs[o]).shape → β → α

namespace AddressedProgramIO
abbrev Inputs {α β : Type} (io : AddressedProgramIO α β) := ChannelValues α io.inputs
abbrev Outputs {α β : Type} (io : AddressedProgramIO α β) := ChannelValues α io.outputs

def InBounds {α β : Type} (io : AddressedProgramIO α β) (bounds : RegionName → Nat)
    (vals : io.Inputs) : Prop :=
  (∀ r, r ∉ io.buffers → bounds r = 0) ∧
  (∀ i j, io.inputMask i vals j → io.read i vals j < bounds io.inputs[i].buf) ∧
  (∀ o j, io.outputMask o vals j → io.write o vals j < bounds io.outputs[o].buf) ∧
  (∀ k j, io.scratchMask k vals j → io.scratchWrite k vals j < bounds io.scratch[k].buf)

/-- Encode only floating payloads at the actual declared read region/offset.
Typed read/default behavior and discrete metadata match ProgramIO.InputBinding.
-/
def InputBinding {α β : Type} [Inhabited β] (io : AddressedProgramIO α β)
    (vals : io.Inputs) (s : State β) : Prop :=
  ∀ i j, io.inputMask i vals j →
    (s.mem io.inputs[i].buf (io.read i vals j)).read io.inputs[i].dtype =
      mapValue (io.encodeAt io.inputs[i].buf (io.read i vals j)) io.inputs[i].dtype (vals i j)

def OutputBinding {α β : Type} (io : AddressedProgramIO α β) (vals : io.Inputs)
    (expected : io.Outputs) (s : State β) : Prop :=
  ∀ o j, io.outputMask o vals j →
    mapCell (io.observe o j) (s.mem io.outputs[o].buf (io.write o vals j)) =
      Cell.mk io.outputs[o].dtype (expected o j)

def Outside {α β : Type} (io : AddressedProgramIO α β) (vals : io.Inputs)
    (r : RegionName) (offset : Nat) : Prop :=
  (∀ o j, io.outputMask o vals j → r ≠ io.outputs[o].buf ∨ offset ≠ io.write o vals j) ∧
  (∀ k j, io.scratchMask k vals j → r ≠ io.scratch[k].buf ∨ offset ≠ io.scratchWrite k vals j)

def Frame {α β : Type} (io : AddressedProgramIO α β) (vals : io.Inputs)
    (before after : State β) : Prop :=
  ∀ r offset, io.Outside vals r offset → after.mem r offset = before.mem r offset

/-- Actual safe, successful source execution with independent logical outputs.
Address labels affect input representation only, never the source runner/frame.
-/
def ImplementsOn {α β : Type} [Inhabited β] (io : AddressedProgramIO α β)
    (domain : io.Inputs → Prop) (f : io.Inputs → io.Outputs) : Prop :=
  ∀ vals, domain vals → ∀ bounds, io.InBounds bounds vals → ∀ initial,
    io.InputBinding vals initial →
    ScheduleSafe bounds (io.schedule vals) initial ∧
    ∃ final, runSchedule (io.schedule vals) initial = some final ∧
      io.OutputBinding vals (f vals) final ∧ io.Frame vals initial final

def Implements {α β : Type} [Inhabited β] (io : AddressedProgramIO α β)
    (f : io.Inputs → io.Outputs) : Prop := io.ImplementsOn (fun _ => True) f

scoped infix:50 " ⊨ " => Implements

theorem ImplementsOn.intro {α β : Type} [Inhabited β] {io : AddressedProgramIO α β}
    {domain : io.Inputs → Prop} {f : io.Inputs → io.Outputs}
    (safe : ∀ vals, domain vals → ∀ bounds, io.InBounds bounds vals → ∀ initial,
      io.InputBinding vals initial → ScheduleSafe bounds (io.schedule vals) initial)
    (correct : ∀ vals, domain vals → ∀ bounds, io.InBounds bounds vals → ∀ initial,
      io.InputBinding vals initial →
      ∃ final, runSchedule (io.schedule vals) initial = some final ∧
        io.OutputBinding vals (f vals) final ∧ io.Frame vals initial final) :
    io.ImplementsOn domain f := by
  intro vals domain bounds bounded initial inputs
  exact ⟨safe vals domain bounds bounded initial inputs,
    correct vals domain bounds bounded initial inputs⟩

theorem Implements.intro {α β : Type} [Inhabited β] {io : AddressedProgramIO α β}
    {f : io.Inputs → io.Outputs}
    (safe : ∀ vals bounds, io.InBounds bounds vals → ∀ initial,
      io.InputBinding vals initial → ScheduleSafe bounds (io.schedule vals) initial)
    (correct : ∀ vals initial, io.InputBinding vals initial →
      ∃ final, runSchedule (io.schedule vals) initial = some final ∧
        io.OutputBinding vals (f vals) final ∧ io.Frame vals initial final) :
    io.Implements f := by
  intro vals _ bounds bounded initial inputs
  exact ⟨safe vals bounds bounded initial inputs, correct vals initial inputs⟩

/-- Uniform encoders are the address-independent special case. -/
def ofProgramIO {α β : Type} (io : ProgramIO α β) : AddressedProgramIO α β where
  toMemoryIO := io.toMemoryIO
  schedule := io.schedule
  encodeAt := fun _ _ => io.encode
  observe := io.observe

/-- Embedding the existing interface changes none of its proof obligations. -/
theorem ofProgramIO_implementsOn_iff {α β : Type} [Inhabited β] (io : ProgramIO α β)
    (domain : io.Inputs → Prop) (f : io.Inputs → io.Outputs) :
    (ofProgramIO io).ImplementsOn domain f ↔ io.ImplementsOn domain f := Iff.rfl

theorem ofProgramIO_implements_iff {α β : Type} [Inhabited β] (io : ProgramIO α β)
    (f : io.Inputs → io.Outputs) :
    (ofProgramIO io).Implements f ↔ io.Implements f := Iff.rfl

end AddressedProgramIO
end
end VeriTile.Triton.FP.Structural
