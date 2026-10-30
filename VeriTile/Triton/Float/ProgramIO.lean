import VeriTile.Triton.Float.StructuralSafety

/-! Typed regional IO for actual Structural kernels and finite launch schedules.

This interface fixes execution to `Structural.exec`. A launch may bind runtime
register arguments and choose an algebra, but cannot supply its own execution
or safety predicate. The judgment proves source trace safety, termination,
typed logical outputs, and an exact cell frame. It makes no flat-pointer
transport claim: that remains a separate memory-model theorem.
-/
namespace VeriTile.Triton.FP.Structural
noncomputable section

/-- Map floating payloads only; preserve discrete values and their types. -/
def mapValue {α β : Type} (f : α → β) : (d : TileDType) → Value α d → Value β d
  | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 => f
  | .nat | .int | .bool | .ptr | .blockPtr => id

def mapCell {α β : Type} (f : α → β) : Cell α → Cell β
  | .mk d v => .mk d (mapValue f d v)

/-- One runtime ABI register binding; this is data, not a state transformer. -/
structure RegisterBinding (α : Type) where
  name : RegName
  dtype : TileDType
  shape : TileShape
  value : Values α dtype shape

def bindRegisters {α : Type} (args : List (RegisterBinding α)) (s : State α) : State α :=
  args.foldl (fun t a => t.setReg a.name a.dtype a.shape a.value) s

@[simp] theorem bindRegisters_mem {α : Type} (args : List (RegisterBinding α))
    (s : State α) : (bindRegisters args s).mem = s.mem := by
  induction args generalizing s with
  | nil => rfl
  | cons a rest ih => exact ih _

/-- A concrete source launch. The numerical backend is the interpreter's
primitive model; binding arguments never reads or writes memory. -/
structure Invocation (α : Type) where
  kernel : ComputeKernel
  backend : Algebra α
  arguments : List (RegisterBinding α) := []
  /-- Some selected launch models start with fresh local registers. -/
  clearRegisters : Bool := false
  pids : Nat → Nat
  /-- Explicitly select the program-ID axes installed by this launch. -/
  setPids : Nat → Bool := fun _ => true
  numPids : Option (Nat → Nat) := none

namespace Invocation
def initial {α : Type} (job : Invocation α) (s : State α) : State α :=
  let pids := fun ax => if job.setPids ax then job.pids ax else s.pids ax
  let base := { s with pids := pids, numPids := job.numPids.getD s.numPids }
  bindRegisters job.arguments
    { base with regs := if job.clearRegisters then fun _ _ _ => none else s.regs }

@[simp] theorem initial_mem {α : Type} (job : Invocation α) (s : State α) :
    (job.initial s).mem = s.mem := bindRegisters_mem _ _

def run {α : Type} [Inhabited α] (job : Invocation α) (s : State α) : Option (State α) :=
  exec job.backend job.kernel (job.initial s)

def Safe {α : Type} [Inhabited α] (job : Invocation α)
    (bounds : RegionName → Nat) (s : State α) : Prop :=
  TraceSafe job.backend bounds job.kernel (job.initial s)
end Invocation

/-- Launch order is explicit. Every successor receives the actual predecessor
memory; per-launch register arguments and program IDs are then installed. -/
def runSchedule {α : Type} [Inhabited α] : List (Invocation α) → State α → Option (State α)
  | [], s => some s
  | job :: rest, s => (job.run s).bind (runSchedule rest)

/-- Safety follows the same execution path as `runSchedule`, including all
intermediate memories. Successful execution is a separate obligation. -/
def ScheduleSafe {α : Type} [Inhabited α] (bounds : RegionName → Nat) :
    List (Invocation α) → State α → Prop
  | [], _ => True
  | job :: rest, s => job.Safe bounds s ∧
      ∀ t, job.run s = some t → ScheduleSafe bounds rest t

@[simp] theorem runSchedule_singleton {α : Type} [Inhabited α]
    (job : Invocation α) (s : State α) : runSchedule [job] s = job.run s := by
  cases h : job.run s <;> simp [runSchedule, h]

@[simp] theorem scheduleSafe_singleton {α : Type} [Inhabited α]
    (bounds : RegionName → Nat) (job : Invocation α) (s : State α) :
    ScheduleSafe bounds [job] s ↔ job.Safe bounds s := by
  simp [ScheduleSafe]

theorem runSchedule_append {α : Type} [Inhabited α]
    (xs ys : List (Invocation α)) (s : State α) :
    runSchedule (xs ++ ys) s = (runSchedule xs s).bind (runSchedule ys) := by
  induction xs generalizing s with
  | nil => rfl
  | cons job rest ih =>
      simp only [List.cons_append, runSchedule]
      cases h : job.run s <;> simp [ih]

theorem scheduleSafe_append {α : Type} [Inhabited α] (bounds : RegionName → Nat)
    (xs ys : List (Invocation α)) (s : State α) :
    ScheduleSafe bounds (xs ++ ys) s ↔ ScheduleSafe bounds xs s ∧
      ∀ t, runSchedule xs s = some t → ScheduleSafe bounds ys t := by
  induction xs generalizing s with
  | nil => simp [ScheduleSafe, runSchedule]
  | cons job rest ih =>
      simp only [List.cons_append, ScheduleSafe, runSchedule]
      cases h : job.run s <;> simp [ih, and_assoc]

/-- A logical channel's shape, interpreter value type, and region. Source
storage/computation precision remains fixed by each invocation's source AST;
`dtype` names the actual Structural cell channel (which can be erased Real).
-/
structure IOChannel where
  dtype : TileDType
  shape : TileShape
  buf : RegionName

abbrev ChannelValues (α : Type) (ports : List IOChannel) :=
  (i : Fin ports.length) → Values α (ports[i]).dtype (ports[i]).shape

/-- Public regional memory layout. Windows may depend on logical metadata
channels; every active channel is pinned simultaneously in the initial memory.
Private scratch is explicitly listed and excluded only at its active offsets.
-/
structure MemoryIO (α : Type) where
  inputs : List IOChannel
  outputs : List IOChannel
  scratch : List IOChannel := []
  buffers : List RegionName
  read : (i : Fin inputs.length) → ChannelValues α inputs → TileIndex (inputs[i]).shape → Nat
  inputMask : (i : Fin inputs.length) → ChannelValues α inputs → TileIndex (inputs[i]).shape → Prop
  write : (o : Fin outputs.length) → ChannelValues α inputs → TileIndex (outputs[o]).shape → Nat
  outputMask : (o : Fin outputs.length) → ChannelValues α inputs → TileIndex (outputs[o]).shape → Prop
  scratchWrite : (k : Fin scratch.length) → ChannelValues α inputs → TileIndex (scratch[k]).shape → Nat
  scratchMask : (k : Fin scratch.length) → ChannelValues α inputs → TileIndex (scratch[k]).shape → Prop

/-- Source-connected IO with a logical carrier and an execution carrier.
Only floating inputs/outputs are encoded/observed. Output-specific observers
support native vector lanes without changing discrete values or dtype tags.
-/
structure ProgramIO (α β : Type) extends MemoryIO α where
  schedule : ChannelValues α inputs → List (Invocation β)
  encode : α → β
  observe : (o : Fin outputs.length) → TileIndex (outputs[o]).shape → β → α

namespace ProgramIO
abbrev Inputs {α β : Type} (io : ProgramIO α β) := ChannelValues α io.inputs
abbrev Outputs {α β : Type} (io : ProgramIO α β) := ChannelValues α io.outputs

def InBounds {α β : Type} (io : ProgramIO α β) (bounds : RegionName → Nat)
    (vals : io.Inputs) : Prop :=
  (∀ r, r ∉ io.buffers → bounds r = 0) ∧
  (∀ i j, io.inputMask i vals j → io.read i vals j < bounds io.inputs[i].buf) ∧
  (∀ o j, io.outputMask o vals j → io.write o vals j < bounds io.outputs[o].buf) ∧
  (∀ k j, io.scratchMask k vals j → io.scratchWrite k vals j < bounds io.scratch[k].buf)

def InputBinding {α β : Type} [Inhabited β] (io : ProgramIO α β)
    (vals : io.Inputs) (s : State β) : Prop :=
  ∀ i j, io.inputMask i vals j →
    (s.mem io.inputs[i].buf (io.read i vals j)).read io.inputs[i].dtype =
      mapValue io.encode io.inputs[i].dtype (vals i j)

def OutputBinding {α β : Type} (io : ProgramIO α β) (vals : io.Inputs)
    (expected : io.Outputs) (s : State β) : Prop :=
  ∀ o j, io.outputMask o vals j →
    mapCell (io.observe o j) (s.mem io.outputs[o].buf (io.write o vals j)) =
      Cell.mk io.outputs[o].dtype (expected o j)

def Outside {α β : Type} (io : ProgramIO α β) (vals : io.Inputs)
    (r : RegionName) (offset : Nat) : Prop :=
  (∀ o j, io.outputMask o vals j → r ≠ io.outputs[o].buf ∨ offset ≠ io.write o vals j) ∧
  (∀ k j, io.scratchMask k vals j → r ≠ io.scratch[k].buf ∨ offset ≠ io.scratchWrite k vals j)

def Frame {α β : Type} (io : ProgramIO α β) (vals : io.Inputs)
    (before after : State β) : Prop :=
  ∀ r offset, io.Outside vals r offset → after.mem r offset = before.mem r offset

/-- Independent logical-domain conditions are explicit arguments of the shared
judgment. Source access safety is always a conclusion, never a domain field.
-/
def ImplementsOn {α β : Type} [Inhabited β] (io : ProgramIO α β)
    (domain : io.Inputs → Prop) (f : io.Inputs → io.Outputs) : Prop :=
  ∀ vals, domain vals → ∀ bounds, io.InBounds bounds vals → ∀ initial,
    io.InputBinding vals initial →
    ScheduleSafe bounds (io.schedule vals) initial ∧
    ∃ final, runSchedule (io.schedule vals) initial = some final ∧
      io.OutputBinding vals (f vals) final ∧ io.Frame vals initial final

def Implements {α β : Type} [Inhabited β] (io : ProgramIO α β)
    (f : io.Inputs → io.Outputs) : Prop := io.ImplementsOn (fun _ => True) f

scoped infix:50 " ⊨ " => Implements
scoped notation:25 io " ⊨{" domain "} " f => ImplementsOn io domain f

theorem Implements.intro {α β : Type} [Inhabited β] {io : ProgramIO α β}
    {f : io.Inputs → io.Outputs}
    (safe : ∀ vals bounds, io.InBounds bounds vals → ∀ initial,
      io.InputBinding vals initial → ScheduleSafe bounds (io.schedule vals) initial)
    (correct : ∀ vals initial, io.InputBinding vals initial →
      ∃ final, runSchedule (io.schedule vals) initial = some final ∧
        io.OutputBinding vals (f vals) final ∧ io.Frame vals initial final) : io.Implements f := by
  intro vals _ bounds hb initial hin
  exact ⟨safe vals bounds hb initial hin, correct vals initial hin⟩

end ProgramIO
end
end VeriTile.Triton.FP.Structural
