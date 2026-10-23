import VeriTile.Triton.Float.ProgramIO

/-! Read-view correctness with separate result and writable memory windows.

An in-place kernel may report a value from every logical result lane while
writing only a strict subset. Unwritten cells keep their exact original tags;
read-view specifications therefore use the evaluator's declared typed read,
including its existing default on a mismatched tag. This interface preserves
that input domain without allowing the larger result window to weaken the
exact writable-cell frame. Execution and safety still use actual invocations.
-/
namespace VeriTile.Triton.FP.Structural
noncomputable section

/-- The same typed channels and concrete execution as `ProgramIO`, with an
independent result-read window and mask. `write`/`outputMask` continue to name
only writable cells; result reads may also observe untouched cells. -/
structure ReadProgramIO (α β : Type) extends ProgramIO α β where
  resultRead : (o : Fin outputs.length) → ChannelValues α inputs →
    TileIndex (outputs[o]).shape → Nat
  resultMask : (o : Fin outputs.length) → ChannelValues α inputs →
    TileIndex (outputs[o]).shape → Prop

namespace ReadProgramIO
abbrev Inputs {α β : Type} (io : ReadProgramIO α β) := io.toProgramIO.Inputs
abbrev Outputs {α β : Type} (io : ReadProgramIO α β) := io.toProgramIO.Outputs

/-- All source input/write/scratch windows and all observed result reads are
backed. Result-only cells do not become writable through this premise. -/
def InBounds {α β : Type} (io : ReadProgramIO α β) (bounds : RegionName → Nat)
    (vals : io.Inputs) : Prop :=
  io.toProgramIO.InBounds bounds vals ∧
  ∀ o j, io.resultMask o vals j → io.resultRead o vals j < bounds io.outputs[o].buf

/-- Preserve exactly the original evaluator input-read convention. -/
abbrev InputBinding {α β : Type} [Inhabited β] (io : ReadProgramIO α β)
    (vals : io.Inputs) (s : State β) : Prop := io.toProgramIO.InputBinding vals s

/-- Decode the specified read dtype before observing a floating payload.
No new stored-tag premise is imposed on an unchanged result cell. -/
def OutputReadback {α β : Type} [Inhabited β] (io : ReadProgramIO α β)
    (vals : io.Inputs) (expected : io.Outputs) (s : State β) : Prop :=
  ∀ o j, io.resultMask o vals j →
    mapValue (io.observe o j) io.outputs[o].dtype
      ((s.mem io.outputs[o].buf (io.resultRead o vals j)).read io.outputs[o].dtype) =
      expected o j

/-- The exact raw frame uses writable windows, never the larger result windows. -/
abbrev Frame {α β : Type} (io : ReadProgramIO α β) (vals : io.Inputs)
    (before after : State β) : Prop := io.toProgramIO.Frame vals before after

def ImplementsOn {α β : Type} [Inhabited β] (io : ReadProgramIO α β)
    (domain : io.Inputs → Prop) (f : io.Inputs → io.Outputs) : Prop :=
  ∀ vals, domain vals → ∀ bounds, io.InBounds bounds vals → ∀ initial,
    io.InputBinding vals initial →
    ScheduleSafe bounds (io.schedule vals) initial ∧
    ∃ final, runSchedule (io.schedule vals) initial = some final ∧
      io.OutputReadback vals (f vals) final ∧ io.Frame vals initial final

def Implements {α β : Type} [Inhabited β] (io : ReadProgramIO α β)
    (f : io.Inputs → io.Outputs) : Prop := io.ImplementsOn (fun _ => True) f

scoped infix:50 " ⊨ " => Implements
scoped notation:25 io " ⊨{" domain "} " f => ImplementsOn io domain f

/-- Assemble concrete source safety and the existing read-value/frame proof. -/
theorem ImplementsOn.intro {α β : Type} [Inhabited β] {io : ReadProgramIO α β}
    {domain : io.Inputs → Prop} {f : io.Inputs → io.Outputs}
    (safe : ∀ vals, domain vals → ∀ bounds, io.InBounds bounds vals →
      ∀ initial, io.InputBinding vals initial →
        ScheduleSafe bounds (io.schedule vals) initial)
    (correct : ∀ vals, domain vals → ∀ bounds, io.InBounds bounds vals →
      ∀ initial, io.InputBinding vals initial →
        ∃ final, runSchedule (io.schedule vals) initial = some final ∧
          io.OutputReadback vals (f vals) final ∧ io.Frame vals initial final) :
    io.ImplementsOn domain f := by
  intro vals hd bounds hb initial hin
  exact ⟨safe vals hd bounds hb initial hin, correct vals hd bounds hb initial hin⟩

end ReadProgramIO
end
end VeriTile.Triton.FP.Structural
