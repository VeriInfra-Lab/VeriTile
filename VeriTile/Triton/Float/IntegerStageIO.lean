import VeriTile.Triton.Float.IntegerStageSafety
import VeriTile.Triton.Float.MemoryIOJudgments

/-! Typed IO for the fixed existing llrint64/integer-store source executor.
Each invocation is concrete source, primitive data and register/PID ABI. The
only execution path is QuantizeIntegerStage.exec. No execution or safety
callback is accepted. Launch lists may repeat programs or change tile sizes;
successors always receive the actual predecessor state.
-/
namespace VeriTile.Triton.FP.Structural
noncomputable section

/-- The ordinary launch ABI plus the source's integer storage signature,
register-marker bindings and optional llrint64 primitive. -/
structure IntegerInvocation (α : Type) extends Invocation α where
  integerStorage : QuantizeIntegerStage.IntegerStorage
  runtimeRegisters : QuantizeIntegerStage.RuntimeRegisters
  round : Option (QuantizeIntegerStage.Llrint64 α)

namespace IntegerInvocation
def typedKernel {α : Type} (job : IntegerInvocation α) : QuantizeIntegerStage.TypedKernel :=
  ⟨job.kernel, job.integerStorage, job.runtimeRegisters⟩

def initial {α : Type} (job : IntegerInvocation α) (s : State α) : State α :=
  job.toInvocation.initial s

@[simp] theorem initial_mem {α : Type} (job : IntegerInvocation α) (s : State α) :
    (job.initial s).mem = s.mem := Invocation.initial_mem _ _

def run {α : Type} [Inhabited α] (job : IntegerInvocation α) (s : State α) : Option (State α) :=
  QuantizeIntegerStage.exec job.backend job.round job.typedKernel (job.initial s)

def Safe {α : Type} [Inhabited α] (job : IntegerInvocation α)
    (bounds : RegionName → Nat) (s : State α) : Prop :=
  QuantizeIntegerStage.TraceSafe job.backend job.round bounds job.typedKernel (job.initial s)
end IntegerInvocation

def runIntegerSchedule {α : Type} [Inhabited α] :
    List (IntegerInvocation α) → State α → Option (State α)
  | [], s => some s
  | job :: rest, s => (job.run s).bind (runIntegerSchedule rest)

def IntegerScheduleSafe {α : Type} [Inhabited α] (bounds : RegionName → Nat) :
    List (IntegerInvocation α) → State α → Prop
  | [], _ => True
  | job :: rest, s => job.Safe bounds s ∧
      ∀ t, job.run s = some t → IntegerScheduleSafe bounds rest t

@[simp] theorem runIntegerSchedule_singleton {α : Type} [Inhabited α]
    (job : IntegerInvocation α) (s : State α) : runIntegerSchedule [job] s = job.run s := by
  cases h : job.run s <;> simp [runIntegerSchedule, h]

@[simp] theorem integerScheduleSafe_singleton {α : Type} [Inhabited α]
    (bounds : RegionName → Nat) (job : IntegerInvocation α) (s : State α) :
    IntegerScheduleSafe bounds [job] s ↔ job.Safe bounds s := by
  simp [IntegerScheduleSafe]

theorem runIntegerSchedule_append {α : Type} [Inhabited α]
    (left right : List (IntegerInvocation α)) (s : State α) :
    runIntegerSchedule (left ++ right) s =
      (runIntegerSchedule left s).bind (runIntegerSchedule right) := by
  induction left generalizing s with
  | nil => rfl
  | cons job rest ih =>
    cases h : job.run s <;> simp [runIntegerSchedule, h, ih]

theorem integerScheduleSafe_append {α : Type} [Inhabited α] (bounds : RegionName → Nat)
    (left right : List (IntegerInvocation α)) (s : State α) :
    IntegerScheduleSafe bounds (left ++ right) s ↔ IntegerScheduleSafe bounds left s ∧
      ∀ t, runIntegerSchedule left s = some t → IntegerScheduleSafe bounds right t := by
  induction left generalizing s with
  | nil => simp [IntegerScheduleSafe, runIntegerSchedule]
  | cons job rest ih =>
    simp only [List.cons_append, IntegerScheduleSafe, runIntegerSchedule]
    cases h : job.run s <;> simp [ih, and_assoc]

/-- Same typed memory interface as ProgramIO, connected to the fixed existing
integer-stage executor. Native integer outputs retain their exact signed cells. -/
structure IntegerProgramIO (α β : Type) extends MemoryIO α where
  schedule : ChannelValues α inputs → List (IntegerInvocation β)
  encode : α → β
  observe : (o : Fin outputs.length) → TileIndex (outputs[o]).shape → β → α

namespace IntegerProgramIO
abbrev Inputs {α β : Type} (io : IntegerProgramIO α β) := ChannelValues α io.inputs
abbrev Outputs {α β : Type} (io : IntegerProgramIO α β) := ChannelValues α io.outputs

abbrev InBounds {α β : Type} (io : IntegerProgramIO α β) := io.toMemoryIO.InBounds
abbrev InputBinding {α β : Type} [Inhabited β] (io : IntegerProgramIO α β) :=
  io.toMemoryIO.InputBinding io.encode
abbrev OutputBinding {α β : Type} (io : IntegerProgramIO α β) :=
  io.toMemoryIO.OutputBinding io.observe
abbrev Outside {α β : Type} (io : IntegerProgramIO α β) := io.toMemoryIO.Outside
abbrev Frame {α β : Type} (io : IntegerProgramIO α β) := @MemoryIO.Frame α β io.toMemoryIO

def ImplementsOn {α β : Type} [Inhabited β] (io : IntegerProgramIO α β)
    (domain : io.Inputs → Prop) (f : io.Inputs → io.Outputs) : Prop :=
  ∀ vals, domain vals → ∀ bounds, io.InBounds bounds vals → ∀ initial,
    io.InputBinding vals initial →
    IntegerScheduleSafe bounds (io.schedule vals) initial ∧
    ∃ final, runIntegerSchedule (io.schedule vals) initial = some final ∧
      io.OutputBinding vals (f vals) final ∧ io.Frame vals initial final

def Implements {α β : Type} [Inhabited β] (io : IntegerProgramIO α β)
    (f : io.Inputs → io.Outputs) : Prop := io.ImplementsOn (fun _ => True) f

scoped infix:50 " ⊨ " => Implements

theorem Implements.intro {α β : Type} [Inhabited β] {io : IntegerProgramIO α β}
    {f : io.Inputs → io.Outputs}
    (safe : ∀ vals bounds, io.InBounds bounds vals → ∀ initial,
      io.InputBinding vals initial → IntegerScheduleSafe bounds (io.schedule vals) initial)
    (correct : ∀ vals initial, io.InputBinding vals initial →
      ∃ final, runIntegerSchedule (io.schedule vals) initial = some final ∧
        io.OutputBinding vals (f vals) final ∧ io.Frame vals initial final) : io.Implements f := by
  intro vals _ bounds hb initial hin
  exact ⟨safe vals bounds hb initial hin, correct vals initial hin⟩

/-- Another concrete source schedule on exactly the same typed memory layout. -/
structure Variant {α β : Type} (io : IntegerProgramIO α β) where
  schedule : io.Inputs → List (IntegerInvocation β)
  observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → β → α

def Variant.toIntegerProgramIO {α β : Type} {io : IntegerProgramIO α β}
    (candidate : Variant io) : IntegerProgramIO α β :=
  { io with schedule := candidate.schedule, observe := candidate.observe }

def OutputsEqual {α β : Type} (io : IntegerProgramIO α β) (candidate : Variant io)
    (vals : io.Inputs) (left right : State β) : Prop :=
  ∀ o j, io.outputMask o vals j →
    mapCell (io.observe o j) (left.mem io.outputs[o].buf (io.write o vals j)) =
      mapCell (candidate.observe o j) (right.mem io.outputs[o].buf (io.write o vals j))

/-- Both source schedules succeed safely and agree on complete typed outputs,
with both exact exterior frames. An identical failure is not equivalence. -/
def EquivalentOn {α β : Type} [Inhabited β] (io : IntegerProgramIO α β)
    (candidate : Variant io) (domain : io.Inputs → Prop) : Prop :=
  ∀ vals, domain vals → ∀ bounds, io.InBounds bounds vals → ∀ initial,
    io.InputBinding vals initial →
    IntegerScheduleSafe bounds (io.schedule vals) initial ∧
    IntegerScheduleSafe bounds (candidate.schedule vals) initial ∧
    ∃ left right,
      runIntegerSchedule (io.schedule vals) initial = some left ∧
      runIntegerSchedule (candidate.schedule vals) initial = some right ∧
      io.OutputsEqual candidate vals left right ∧
      io.Frame vals initial left ∧ io.Frame vals initial right

def Equivalent {α β : Type} [Inhabited β] (io : IntegerProgramIO α β)
    (candidate : Variant io) : Prop := io.EquivalentOn candidate (fun _ => True)

theorem EquivalentOn.of_implementsOn {α β : Type} [Inhabited β]
    {io : IntegerProgramIO α β} {candidate : Variant io}
    {domain : io.Inputs → Prop} {f : io.Inputs → io.Outputs}
    (left : io.ImplementsOn domain f)
    (right : candidate.toIntegerProgramIO.ImplementsOn domain f) :
    io.EquivalentOn candidate domain := by
  intro vals hd bounds hb initial hin
  obtain ⟨leftSafe, l, leftRun, leftOut, leftFrame⟩ := left vals hd bounds hb initial hin
  obtain ⟨rightSafe, r, rightRun, rightOut, rightFrame⟩ := right vals hd bounds hb initial hin
  refine ⟨leftSafe, rightSafe, l, r, leftRun, rightRun, ?_, leftFrame, rightFrame⟩
  intro o j active
  exact (leftOut o j active).trans (rightOut o j active).symm

end IntegerProgramIO
end
end VeriTile.Triton.FP.Structural
