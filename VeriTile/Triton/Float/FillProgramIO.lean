import VeriTile.Triton.Float.ProgramIO

/-! Actual Structural source invocations interleaved with finite constant
typed-cell fills. The fill is closed data and preserves every other cell.
It neither retags existing cells nor replaces source execution with callbacks.
-/
namespace VeriTile.Triton.FP.Structural
noncomputable section

structure FiniteFill (α : Type) where
  region : RegionName
  start : Nat
  count : Nat
  value : Cell α

namespace FiniteFill
def Contains {α : Type} (fill : FiniteFill α) (r : RegionName) (address : Nat) : Prop :=
  r=fill.region ∧ fill.start ≤ address ∧ address<fill.start+fill.count

instance {α : Type} (fill : FiniteFill α) (r : RegionName) (address : Nat) : Decidable (fill.Contains r address) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _))

def apply {α : Type} (fill : FiniteFill α) (s : State α) : State α :=
  {s with mem:=fun r address=>if fill.Contains r address then fill.value else s.mem r address}

def Safe {α : Type} (fill : FiniteFill α) (bounds : RegionName → Nat) : Prop :=
  ∀ i,i<fill.count → fill.start+i<bounds fill.region

@[simp] theorem apply_mem_inside {α : Type} (fill : FiniteFill α) (s : State α)
    (i : Nat) (hi : i<fill.count) :
    (fill.apply s).mem fill.region (fill.start+i)=fill.value := by simp [apply,Contains,hi]

theorem apply_mem_outside {α : Type} (fill : FiniteFill α) (s : State α)
    (r : RegionName) (address : Nat) (outside : ¬fill.Contains r address) :
    (fill.apply s).mem r address=s.mem r address := by simp [apply,outside]

@[simp] theorem apply_zero {α : Type} (region : RegionName) (start : Nat) (value : Cell α)
    (s : State α) : (FiniteFill.mk region start 0 value).apply s=s := by
  have empty (r : RegionName) (address : Nat) :
      ¬(FiniteFill.mk region start 0 value).Contains r address := by
    simp only [Contains,Nat.add_zero]
    omega
  simp [apply,empty]
end FiniteFill

inductive FillStage (α : Type) where
  | source (invocation : Invocation α)
  | fill (initialization : FiniteFill α)

namespace FillStage
def run {α : Type} [Inhabited α] : FillStage α → State α → Option (State α)
  | .source invocation,s => invocation.run s
  | .fill initialization,s => some (initialization.apply s)

def Safe {α : Type} [Inhabited α] (bounds : RegionName → Nat) : FillStage α → State α → Prop
  | .source invocation,s => invocation.Safe bounds s
  | .fill initialization,_ => initialization.Safe bounds
end FillStage

def runFillSchedule {α : Type} [Inhabited α] : List (FillStage α) → State α → Option (State α)
  | [],s => some s
  | stage::rest,s => (stage.run s).bind (runFillSchedule rest)

def FillScheduleSafe {α : Type} [Inhabited α] (bounds : RegionName → Nat) :
    List (FillStage α) → State α → Prop
  | [],_ => True
  | stage::rest,s => stage.Safe bounds s ∧
      ∀ t,stage.run s=some t → FillScheduleSafe bounds rest t

theorem runFillSchedule_sources {α : Type} [Inhabited α]
    (jobs : List (Invocation α)) (s : State α) :
    runFillSchedule (jobs.map FillStage.source) s=runSchedule jobs s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons job rest ih =>
    simp only [List.map_cons,runFillSchedule,FillStage.run,runSchedule]
    cases job.run s <;> simp [ih]

theorem fillScheduleSafe_sources {α : Type} [Inhabited α] (bounds : RegionName → Nat)
    (jobs : List (Invocation α)) (s : State α) :
    FillScheduleSafe bounds (jobs.map FillStage.source) s ↔ ScheduleSafe bounds jobs s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons job rest ih => simp only [List.map_cons,FillScheduleSafe,FillStage.Safe,FillStage.run,ScheduleSafe,ih]

structure FillProgramIO (α β : Type) extends MemoryIO α where
  stages : ChannelValues α inputs → List (FillStage β)
  encode : α → β
  observe : (o : Fin outputs.length) → TileIndex (outputs[o]).shape → β → α

namespace FillProgramIO
abbrev Inputs {α β : Type} (io : FillProgramIO α β) := ChannelValues α io.inputs
abbrev Outputs {α β : Type} (io : FillProgramIO α β) := ChannelValues α io.outputs

/-- A port-only view reuses exactly the established memory predicates. Its
empty schedule is never used as this program's execution or safety judgment. -/
def ports {α β : Type} (io : FillProgramIO α β) : ProgramIO α β :=
  {io.toMemoryIO with schedule:=fun _=>[],encode:=io.encode,observe:=io.observe}

abbrev InBounds {α β : Type} (io : FillProgramIO α β) := io.ports.InBounds
abbrev InputBinding {α β : Type} [Inhabited β] (io : FillProgramIO α β) := io.ports.InputBinding
abbrev OutputBinding {α β : Type} (io : FillProgramIO α β) := io.ports.OutputBinding
abbrev Outside {α β : Type} (io : FillProgramIO α β) := io.ports.Outside
abbrev Frame {α β : Type} (io : FillProgramIO α β) := io.ports.Frame

/-- The same source/fill schedule must succeed and establish exact typed
outputs and exterior frame for every bound input. Finite bounds only imply
actual source/fill safety; they cannot make success or correctness vacuous. -/
def ImplementsOn {α β : Type} [Inhabited β] (io : FillProgramIO α β)
    (domain : io.Inputs → Prop) (f : io.Inputs → io.Outputs) : Prop :=
  ∀ vals,domain vals → ∀ initial,io.InputBinding vals initial →
    (∀ bounds,io.InBounds bounds vals → FillScheduleSafe bounds (io.stages vals) initial) ∧
    ∃ final,runFillSchedule (io.stages vals) initial=some final ∧
      io.OutputBinding vals (f vals) final ∧ io.Frame vals initial final

def Implements {α β : Type} [Inhabited β] (io : FillProgramIO α β)
    (f : io.Inputs → io.Outputs) : Prop := io.ImplementsOn (fun _=>True) f
end FillProgramIO
end
end VeriTile.Triton.FP.Structural
