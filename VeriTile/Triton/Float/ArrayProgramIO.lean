import VeriTile.Triton.Float.ProgramIO

/-! Named typed input arrays with finite source access footprints.

Some established raw-source results admit arbitrary arrays and alias-dependent
gathers, without a finite tensor extent. Inputs here are named typed arrays;
they are not states or execution callbacks. Only the declared finite read
footprint needs backing. Binding an array does not claim that its entire Nat
domain is allocated. Aliased roles are bound simultaneously, using decoded
reads so mismatched tags retain the interpreter's default behavior.

Execution remains the actual Invocation schedule. Safety follows its actual
successor states. Successful execution, outputs and the exact exterior frame
are required independently of capacities; no impossible bound can hide a
missing result. A task must prove that its source-free finite footprint covers
the actual accesses, including gathers changed by earlier aliased writes.
-/
namespace VeriTile.Triton.FP.Structural
noncomputable section

structure ArrayIOChannel where
  dtype : TileDType
  buf : RegionName

abbrev ArrayValues (α : Type) (ports : List ArrayIOChannel) :=
  (i : Fin ports.length) → Nat → Value α (ports[i]).dtype

structure ArrayProgramIO (α β : Type) where
  inputs : List ArrayIOChannel
  outputs : List IOChannel
  scratch : List IOChannel := []
  buffers : List RegionName
  /-- Finite source-free footprint; addresses may depend on logical inputs. -/
  reads : (i : Fin inputs.length) → ArrayValues α inputs → List Nat
  write : (o : Fin outputs.length) → ArrayValues α inputs → TileIndex (outputs[o]).shape → Nat
  outputMask : (o : Fin outputs.length) → ArrayValues α inputs → TileIndex (outputs[o]).shape → Prop
  scratchWrite : (k : Fin scratch.length) → ArrayValues α inputs → TileIndex (scratch[k]).shape → Nat
  scratchMask : (k : Fin scratch.length) → ArrayValues α inputs → TileIndex (scratch[k]).shape → Prop
  schedule : ArrayValues α inputs → List (Invocation β)
  encode : α → β
  observe : (o : Fin outputs.length) → TileIndex (outputs[o]).shape → β → α

namespace ArrayProgramIO
variable {α β γ : Type}

abbrev Inputs (io : ArrayProgramIO α β) := ArrayValues α io.inputs
abbrev Outputs (io : ArrayProgramIO α β) := ChannelValues α io.outputs
abbrev ObservedOutputs (io : ArrayProgramIO α β) (γ : Type) := ChannelValues γ io.outputs

def InBounds (io : ArrayProgramIO α β) (bounds : RegionName → Nat) (vals : io.Inputs) : Prop :=
  (∀ r, r ∉ io.buffers → bounds r = 0) ∧
  (∀ i off, off ∈ io.reads i vals → off < bounds io.inputs[i].buf) ∧
  (∀ o j, io.outputMask o vals j → io.write o vals j < bounds io.outputs[o].buf) ∧
  (∀ k j, io.scratchMask k vals j → io.scratchWrite k vals j < bounds io.scratch[k].buf)

def InputBinding [Inhabited β] (io : ArrayProgramIO α β) (vals : io.Inputs) (s : State β) : Prop :=
  ∀ i off, (s.mem io.inputs[i].buf off).read io.inputs[i].dtype =
    mapValue io.encode io.inputs[i].dtype (vals i off)

def OutputBinding (io : ArrayProgramIO α β) (vals : io.Inputs)
    (expected : io.Outputs) (s : State β) : Prop :=
  ∀ o j, io.outputMask o vals j →
    mapCell (io.observe o j) (s.mem io.outputs[o].buf (io.write o vals j)) =
      Cell.mk io.outputs[o].dtype (expected o j)

def Outside (io : ArrayProgramIO α β) (vals : io.Inputs) (r : RegionName) (off : Nat) : Prop :=
  (∀ o j, io.outputMask o vals j → r ≠ io.outputs[o].buf ∨ off ≠ io.write o vals j) ∧
  (∀ k j, io.scratchMask k vals j → r ≠ io.scratch[k].buf ∨ off ≠ io.scratchWrite k vals j)

def Frame (io : ArrayProgramIO α β) (vals : io.Inputs) (before after : State β) : Prop :=
  ∀ r off, io.Outside vals r off → after.mem r off = before.mem r off

def ImplementsOn [Inhabited β] (io : ArrayProgramIO α β)
    (domain : io.Inputs → Prop) (f : io.Inputs → io.Outputs) : Prop :=
  ∀ vals, domain vals → ∀ initial, io.InputBinding vals initial →
    (∀ bounds, io.InBounds bounds vals → ScheduleSafe bounds (io.schedule vals) initial) ∧
    ∃ final, runSchedule (io.schedule vals) initial = some final ∧
      io.OutputBinding vals (f vals) final ∧ io.Frame vals initial final

def Implements [Inhabited β] (io : ArrayProgramIO α β)
    (f : io.Inputs → io.Outputs) : Prop := io.ImplementsOn (fun _ => True) f

scoped infix:50 " ⊨ " => Implements
scoped notation:25 io " ⊨{" domain "} " f => ImplementsOn io domain f

theorem ImplementsOn.intro [Inhabited β] {io : ArrayProgramIO α β}
    {domain : io.Inputs → Prop} {f : io.Inputs → io.Outputs}
    (safe : ∀ vals, domain vals → ∀ bounds, io.InBounds bounds vals → ∀ initial,
      io.InputBinding vals initial → ScheduleSafe bounds (io.schedule vals) initial)
    (correct : ∀ vals, domain vals → ∀ initial, io.InputBinding vals initial →
      ∃ final, runSchedule (io.schedule vals) initial = some final ∧
        io.OutputBinding vals (f vals) final ∧ io.Frame vals initial final) :
    io.ImplementsOn domain f := by
  intro vals hd initial hin
  exact ⟨fun bounds hb => safe vals hd bounds hb initial hin, correct vals hd initial hin⟩

def mapOutputs (io : ArrayProgramIO α β)
    (observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → α → γ)
    (expected : io.Outputs) : io.ObservedOutputs γ :=
  fun o j => mapValue (observe o j) io.outputs[o].dtype (expected o j)

def ObservedOutputBinding (io : ArrayProgramIO α β)
    (observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → α → γ)
    (vals : io.Inputs) (expected : io.ObservedOutputs γ) (s : State β) : Prop :=
  ∀ o j, io.outputMask o vals j →
    mapCell (observe o j)
      (mapCell (io.observe o j) (s.mem io.outputs[o].buf (io.write o vals j))) =
        Cell.mk io.outputs[o].dtype (expected o j)

def ObservesOn [Inhabited β] (io : ArrayProgramIO α β)
    (observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → α → γ)
    (domain : io.Inputs → Prop) (f : io.Inputs → io.ObservedOutputs γ) : Prop :=
  ∀ vals, domain vals → ∀ initial, io.InputBinding vals initial →
    (∀ bounds, io.InBounds bounds vals → ScheduleSafe bounds (io.schedule vals) initial) ∧
    ∃ final, runSchedule (io.schedule vals) initial = some final ∧
      io.ObservedOutputBinding observe vals (f vals) final ∧ io.Frame vals initial final

def Observes [Inhabited β] (io : ArrayProgramIO α β)
    (observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → α → γ)
    (f : io.Inputs → io.ObservedOutputs γ) : Prop := io.ObservesOn observe (fun _ => True) f

theorem OutputBinding.observe {io : ArrayProgramIO α β} {vals : io.Inputs}
    {expected : io.Outputs} {s : State β} (bound : io.OutputBinding vals expected s)
    (observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → α → γ) :
    io.ObservedOutputBinding observe vals (io.mapOutputs observe expected) s := by
  intro o j active
  exact congrArg (mapCell (observe o j)) (bound o j active)

theorem ObservesOn.of_implementsOn [Inhabited β] {io : ArrayProgramIO α β}
    {domain : io.Inputs → Prop} {f : io.Inputs → io.Outputs}
    (proof : io.ImplementsOn domain f)
    (observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → α → γ) :
    io.ObservesOn observe domain (fun vals => io.mapOutputs observe (f vals)) := by
  intro vals hd initial hin
  obtain ⟨safe, final, ran, output, frame⟩ := proof vals hd initial hin
  exact ⟨safe, final, ran, output.observe observe, frame⟩

theorem Observes.of_implements [Inhabited β] {io : ArrayProgramIO α β}
    {f : io.Inputs → io.Outputs} (proof : io.Implements f)
    (observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → α → γ) :
    io.Observes observe (fun vals => io.mapOutputs observe (f vals)) :=
  ObservesOn.of_implementsOn proof observe

/-- Every raw initial memory has a named-array snapshot, without a tag guard
or an allocation claim for its unused suffix. -/
def snapshot [Inhabited α] (io : ArrayProgramIO α α) (s : State α) : io.Inputs :=
  fun i off => (s.mem io.inputs[i].buf off).read io.inputs[i].dtype

theorem snapshot_binding [Inhabited α] (io : ArrayProgramIO α α)
    (identity : io.encode = id) (s : State α) : io.InputBinding (io.snapshot s) s := by
  intro i off
  simp only [snapshot, identity]
  cases io.inputs[i].dtype <;> rfl

end ArrayProgramIO
end
end VeriTile.Triton.FP.Structural
