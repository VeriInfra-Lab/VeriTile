import VeriTile.Triton.Float.MaterializedInvocation
import VeriTile.Triton.Float.MemoryIOJudgments

/-! Logical IO for fixed per-CTA materialization schedules.

Inputs and outputs are finite typed channels. Active inputs have their exact
declared tags; output observation changes only floating payloads. The exterior
frame compares the actual raw stored cells, including tags and discrete values.
Access safety concerns the actual source invocation after every materialized
successor. It is a conclusion, independent of successful execution and outputs.
-/
namespace VeriTile.Triton.FP.Structural
noncomputable section

/-- Logical, stored, and source-execution payload carriers are independent.
Materialization and source execution are the fixed shared operations above. -/
structure MaterializedProgramIO (α β γ : Type) extends MemoryIO α where
  schedule : ChannelValues α inputs → List (MaterializedInvocation β γ)
  encode : α → β
  observe : (o : Fin outputs.length) → TileIndex (outputs[o]).shape → β → α

namespace MaterializedProgramIO
variable {α β γ : Type}

abbrev Inputs (io : MaterializedProgramIO α β γ) := io.toMemoryIO.Inputs
abbrev Outputs (io : MaterializedProgramIO α β γ) := io.toMemoryIO.Outputs
abbrev InBounds (io : MaterializedProgramIO α β γ) := io.toMemoryIO.InBounds
abbrev Outside (io : MaterializedProgramIO α β γ) := io.toMemoryIO.Outside

/-- Active input tags and payloads are exact; all other initial cells are free. -/
def InputCells (io : MaterializedProgramIO α β γ) (vals : io.Inputs)
    (memory : RegionName → Nat → Cell β) : Prop :=
  ∀ i j, io.inputMask i vals j →
    memory io.inputs[i].buf (io.read i vals j) =
      Cell.mk io.inputs[i].dtype (mapValue io.encode io.inputs[i].dtype (vals i j))

def OutputBinding (io : MaterializedProgramIO α β γ) (vals : io.Inputs)
    (expected : io.Outputs) (memory : RegionName → Nat → Cell β) : Prop :=
  ∀ o j, io.outputMask o vals j →
    mapCell (io.observe o j) (memory io.outputs[o].buf (io.write o vals j)) =
      Cell.mk io.outputs[o].dtype (expected o j)

def Frame (io : MaterializedProgramIO α β γ) (vals : io.Inputs)
    (before after : RegionName → Nat → Cell β) : Prop :=
  ∀ region offset, io.Outside vals region offset → after region offset = before region offset

/-- No backing-bounds witness is needed to assert execution and its result.
Every supplied bounds witness must additionally establish actual source safety. -/
def ImplementsOnCells [Inhabited γ] (io : MaterializedProgramIO α β γ)
    (domain : io.Inputs → Prop) (f : io.Inputs → io.Outputs) : Prop :=
  ∀ vals, domain vals → ∀ initial, io.InputCells vals initial →
    (∀ bounds, io.InBounds bounds vals →
      MaterializedScheduleSafe bounds (io.schedule vals) initial) ∧
    ∃ final, runMaterializedSchedule (io.schedule vals) initial = some final ∧
      io.OutputBinding vals (f vals) final ∧ io.Frame vals initial final

def ImplementsCells [Inhabited γ] (io : MaterializedProgramIO α β γ)
    (f : io.Inputs → io.Outputs) : Prop := io.ImplementsOnCells (fun _ => True) f

theorem ImplementsOnCells.intro [Inhabited γ] {io : MaterializedProgramIO α β γ}
    {domain : io.Inputs → Prop} {f : io.Inputs → io.Outputs}
    (safe : ∀ vals, domain vals → ∀ bounds, io.InBounds bounds vals →
      ∀ initial, io.InputCells vals initial →
        MaterializedScheduleSafe bounds (io.schedule vals) initial)
    (correct : ∀ vals, domain vals → ∀ initial, io.InputCells vals initial →
      ∃ final, runMaterializedSchedule (io.schedule vals) initial = some final ∧
        io.OutputBinding vals (f vals) final ∧ io.Frame vals initial final) :
    io.ImplementsOnCells domain f := by
  intro vals admitted initial inputs
  exact ⟨fun bounds backed => safe vals admitted bounds backed initial inputs,
    correct vals admitted initial inputs⟩

theorem ImplementsCells.intro [Inhabited γ] {io : MaterializedProgramIO α β γ}
    {f : io.Inputs → io.Outputs}
    (safe : ∀ vals bounds, io.InBounds bounds vals → ∀ initial,
      io.InputCells vals initial → MaterializedScheduleSafe bounds (io.schedule vals) initial)
    (correct : ∀ vals initial, io.InputCells vals initial →
      ∃ final, runMaterializedSchedule (io.schedule vals) initial = some final ∧
        io.OutputBinding vals (f vals) final ∧ io.Frame vals initial final) :
    io.ImplementsCells f := by
  intro vals _ initial inputs
  exact ⟨fun bounds backed => safe vals bounds backed initial inputs,
    correct vals initial inputs⟩

end MaterializedProgramIO
end
end VeriTile.Triton.FP.Structural
