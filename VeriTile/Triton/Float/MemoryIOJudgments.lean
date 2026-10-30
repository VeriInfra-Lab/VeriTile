import VeriTile.Triton.Float.ProgramIO

/-! Pure typed-memory judgments shared by fixed source executors.
These definitions use only a declared MemoryIO and explicit payload encoders/
observers. They contain no executor, safety or arbitrary state callback.
The existing ProgramIO judgments are definitionally identical specializations.
-/
namespace VeriTile.Triton.FP.Structural
namespace MemoryIO
noncomputable section

abbrev Inputs {α : Type} (io : MemoryIO α) := ChannelValues α io.inputs
abbrev Outputs {α : Type} (io : MemoryIO α) := ChannelValues α io.outputs

def InBounds {α : Type} (io : MemoryIO α) (bounds : RegionName → Nat)
    (vals : io.Inputs) : Prop :=
  (∀ r, r ∉ io.buffers → bounds r = 0) ∧
  (∀ i j, io.inputMask i vals j → io.read i vals j < bounds io.inputs[i].buf) ∧
  (∀ o j, io.outputMask o vals j → io.write o vals j < bounds io.outputs[o].buf) ∧
  (∀ k j, io.scratchMask k vals j → io.scratchWrite k vals j < bounds io.scratch[k].buf)

def InputBinding {α β : Type} [Inhabited β] (io : MemoryIO α) (encode : α → β)
    (vals : io.Inputs) (s : State β) : Prop :=
  ∀ i j, io.inputMask i vals j →
    (s.mem io.inputs[i].buf (io.read i vals j)).read io.inputs[i].dtype =
      mapValue encode io.inputs[i].dtype (vals i j)

def OutputBinding {α β : Type} (io : MemoryIO α)
    (observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → β → α)
    (vals : io.Inputs) (expected : io.Outputs) (s : State β) : Prop :=
  ∀ o j, io.outputMask o vals j →
    mapCell (observe o j) (s.mem io.outputs[o].buf (io.write o vals j)) =
      Cell.mk io.outputs[o].dtype (expected o j)

def Outside {α : Type} (io : MemoryIO α) (vals : io.Inputs)
    (r : RegionName) (offset : Nat) : Prop :=
  (∀ o j, io.outputMask o vals j → r ≠ io.outputs[o].buf ∨ offset ≠ io.write o vals j) ∧
  (∀ k j, io.scratchMask k vals j → r ≠ io.scratch[k].buf ∨ offset ≠ io.scratchWrite k vals j)

def Frame {α β : Type} (io : MemoryIO α) (vals : io.Inputs)
    (before after : State β) : Prop :=
  ∀ r offset, io.Outside vals r offset → after.mem r offset = before.mem r offset
end
end MemoryIO

namespace ProgramIO
theorem inBounds_memory_iff {α β : Type} (io : ProgramIO α β)
    (bounds : RegionName → Nat) (vals : io.Inputs) :
    io.InBounds bounds vals ↔ io.toMemoryIO.InBounds bounds vals := Iff.rfl

theorem inputBinding_memory_iff {α β : Type} [Inhabited β] (io : ProgramIO α β)
    (vals : io.Inputs) (s : State β) :
    io.InputBinding vals s ↔ io.toMemoryIO.InputBinding io.encode vals s := Iff.rfl

theorem outputBinding_memory_iff {α β : Type} (io : ProgramIO α β)
    (vals : io.Inputs) (expected : io.Outputs) (s : State β) :
    io.OutputBinding vals expected s ↔ io.toMemoryIO.OutputBinding io.observe vals expected s := Iff.rfl

theorem outside_memory_iff {α β : Type} (io : ProgramIO α β)
    (vals : io.Inputs) (r : RegionName) (offset : Nat) :
    io.Outside vals r offset ↔ io.toMemoryIO.Outside vals r offset := Iff.rfl

theorem frame_memory_iff {α β : Type} (io : ProgramIO α β)
    (vals : io.Inputs) (before after : State β) :
    io.Frame vals before after ↔ io.toMemoryIO.Frame vals before after := Iff.rfl
end ProgramIO
end VeriTile.Triton.FP.Structural
