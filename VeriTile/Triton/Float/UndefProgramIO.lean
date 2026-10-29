import VeriTile.Triton.Float.ProgramIO

/-! Explicit logical inputs for the typed fallback values of masked loads.

Fallback inputs name finite typed lanes of the actual initial `undef` function.
They are distinct from memory inputs: a masked-off load is not a memory access,
so these channels introduce no backing-memory capacity requirement. They do
not initialize, replace, or constrain any other part of the execution state.
The schedule, interpreter, source safety, outputs and exact frame remain the
fixed operations of `ProgramIO`.
-/
namespace VeriTile.Triton.FP.Structural
noncomputable section

structure UndefProgramIO (α β : Type) extends ProgramIO α β where
  undefInputs : List IOChannel
  undefRead : (u : Fin undefInputs.length) → ChannelValues α inputs →
    TileIndex (undefInputs[u]).shape → Nat
  undefMask : (u : Fin undefInputs.length) → ChannelValues α inputs →
    TileIndex (undefInputs[u]).shape → Prop

namespace UndefProgramIO

/-- Memory observations and masked-load fallback observations are separate
logical inputs. Neither contains an execution state or a result witness. -/
structure Inputs {α β : Type} (io : UndefProgramIO α β) where
  memory : io.toProgramIO.Inputs
  undefined : ChannelValues α io.undefInputs

abbrev Outputs {α β : Type} (io : UndefProgramIO α β) := io.toProgramIO.Outputs

def InBounds {α β : Type} (io : UndefProgramIO α β) (bounds : RegionName → Nat)
    (vals : io.Inputs) : Prop := io.toProgramIO.InBounds bounds vals.memory

/-- Only the declared active fallback observations are pinned. Discrete
payloads, including signed integer values, are not changed by `encode`. -/
def UndefBinding {α β : Type} (io : UndefProgramIO α β)
    (vals : io.Inputs) (s : State β) : Prop :=
  ∀ u j, io.undefMask u vals.memory j →
    s.undef io.undefInputs[u].dtype io.undefInputs[u].buf
      (io.undefRead u vals.memory j) =
        mapValue io.encode io.undefInputs[u].dtype (vals.undefined u j)

def InputBinding {α β : Type} [Inhabited β] (io : UndefProgramIO α β)
    (vals : io.Inputs) (s : State β) : Prop :=
  io.toProgramIO.InputBinding vals.memory s ∧ io.UndefBinding vals s

def OutputBinding {α β : Type} (io : UndefProgramIO α β) (vals : io.Inputs)
    (expected : io.Outputs) (s : State β) : Prop :=
  io.toProgramIO.OutputBinding vals.memory expected s

def Frame {α β : Type} (io : UndefProgramIO α β) (vals : io.Inputs)
    (before after : State β) : Prop := io.toProgramIO.Frame vals.memory before after

/-- Complete successful execution, typed outputs and the raw cell frame do
not depend on the existence of a backing-memory capacity witness. Declared
physical capacities are used only to establish actual source trace safety. -/
def ImplementsOn {α β : Type} [Inhabited β] (io : UndefProgramIO α β)
    (domain : io.Inputs → Prop) (f : io.Inputs → io.Outputs) : Prop :=
  ∀ vals, domain vals → ∀ initial, io.InputBinding vals initial →
    (∀ bounds, io.InBounds bounds vals →
      ScheduleSafe bounds (io.schedule vals.memory) initial) ∧
    ∃ final, runSchedule (io.schedule vals.memory) initial = some final ∧
      io.OutputBinding vals (f vals) final ∧ io.Frame vals initial final

def Implements {α β : Type} [Inhabited β] (io : UndefProgramIO α β)
    (f : io.Inputs → io.Outputs) : Prop := io.ImplementsOn (fun _ => True) f

/-- An independent output relation may leave the logical answer existential.
Its typed witness must bind to the actual successful source execution. -/
def SatisfiesOn {α β : Type} [Inhabited β] (io : UndefProgramIO α β)
    (domain : io.Inputs → Prop) (predicate : io.Inputs → io.Outputs → Prop) : Prop :=
  ∀ vals, domain vals → ∀ initial, io.InputBinding vals initial →
    (∀ bounds, io.InBounds bounds vals →
      ScheduleSafe bounds (io.schedule vals.memory) initial) ∧
    ∃ final outputs, runSchedule (io.schedule vals.memory) initial = some final ∧
      io.OutputBinding vals outputs final ∧ predicate vals outputs ∧
      io.Frame vals initial final

def Satisfies {α β : Type} [Inhabited β] (io : UndefProgramIO α β)
    (predicate : io.Inputs → io.Outputs → Prop) : Prop :=
  io.SatisfiesOn (fun _ => True) predicate

theorem ImplementsOn.intro {α β : Type} [Inhabited β] {io : UndefProgramIO α β}
    {domain : io.Inputs → Prop} {f : io.Inputs → io.Outputs}
    (safe : ∀ vals, domain vals → ∀ bounds, io.InBounds bounds vals → ∀ initial,
      io.InputBinding vals initial → ScheduleSafe bounds (io.schedule vals.memory) initial)
    (correct : ∀ vals, domain vals → ∀ initial, io.InputBinding vals initial →
      ∃ final, runSchedule (io.schedule vals.memory) initial = some final ∧
        io.OutputBinding vals (f vals) final ∧ io.Frame vals initial final) :
    io.ImplementsOn domain f := by
  intro vals admitted initial inputs
  exact ⟨fun bounds backed => safe vals admitted bounds backed initial inputs,
    correct vals admitted initial inputs⟩

theorem SatisfiesOn.intro {α β : Type} [Inhabited β] {io : UndefProgramIO α β}
    {domain : io.Inputs → Prop} {predicate : io.Inputs → io.Outputs → Prop}
    (safe : ∀ vals, domain vals → ∀ bounds, io.InBounds bounds vals → ∀ initial,
      io.InputBinding vals initial → ScheduleSafe bounds (io.schedule vals.memory) initial)
    (correct : ∀ vals, domain vals → ∀ initial, io.InputBinding vals initial →
      ∃ final outputs, runSchedule (io.schedule vals.memory) initial = some final ∧
        io.OutputBinding vals outputs final ∧ predicate vals outputs ∧
        io.Frame vals initial final) : io.SatisfiesOn domain predicate := by
  intro vals admitted initial inputs
  exact ⟨fun bounds backed => safe vals admitted bounds backed initial inputs,
    correct vals admitted initial inputs⟩

theorem SatisfiesOn.of_implementsOn {α β : Type} [Inhabited β]
    {io : UndefProgramIO α β} {domain : io.Inputs → Prop} {f : io.Inputs → io.Outputs}
    {predicate : io.Inputs → io.Outputs → Prop}
    (implemented : io.ImplementsOn domain f)
    (related : ∀ vals, domain vals → predicate vals (f vals)) :
    io.SatisfiesOn domain predicate := by
  intro vals admitted initial inputs
  obtain ⟨safe, final, ran, outputs, frame⟩ := implemented vals admitted initial inputs
  exact ⟨safe, final, f vals, ran, outputs, related vals admitted, frame⟩

end UndefProgramIO

@[simp] theorem bindRegisters_undef {α : Type} (args : List (RegisterBinding α))
    (s : State α) : (bindRegisters args s).undef = s.undef := by
  induction args generalizing s with
  | nil => rfl
  | cons a rest ih => exact ih _

@[simp] theorem Invocation.initial_undef {α : Type} (job : Invocation α) (s : State α) :
    (job.initial s).undef = s.undef := bindRegisters_undef _ _

end
end VeriTile.Triton.FP.Structural
