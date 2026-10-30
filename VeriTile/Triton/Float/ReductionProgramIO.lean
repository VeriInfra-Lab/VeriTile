import VeriTile.Triton.Float.MemoryIOJudgments

/-! Fixed Structural source stages followed or preceded by finite FP32 sums.

The finite sum is the existing library-operation boundary used by chunk linear
attention. Its plan is a closed zero/input/add tree, not a numerical answer or
an executor callback. Storage uses the existing erased `.real` cell channel:
each leaf applies `fp32Load`, joins use FP32 addition, and the final FP32 cast
is stored with the original `.real` tag. This module makes no claim about
Torch's internal lowering. It records and executes the selected finite plan.
-/
namespace VeriTile.Triton.FP.Structural
noncomputable section

inductive FP32AddTree (n : Nat) where
  | zero
  | input (index : Fin n)
  | add (left right : FP32AddTree n)

namespace FP32AddTree
def leaves {n : Nat} : FP32AddTree n → List (Fin n)
  | .zero => []
  | .input i => [i]
  | .add a b => a.leaves ++ b.leaves

def eval {n : Nat} {α : Type} (M : Algebra α) (values : Fin n → α) : FP32AddTree n → α
  | .zero => M.literal (some .fp32) .real 0
  | .input i => M.fp32Load (values i)
  | .add a b => M.binary (some .fp32) .real .add (a.eval M values) (b.eval M values)
end FP32AddTree

/-- Each selected input occurs exactly once; zero initialization nodes and
the full parenthesization are preserved as part of the numerical plan. -/
structure FP32AddPlan (n : Nat) where
  tree : FP32AddTree n
  coverage : tree.leaves.Perm (List.ofFn (fun i : Fin n => i))

structure FP32SumStage (α : Type) where
  backend : Algebra α
  inputCount : Nat
  outputCount : Nat
  read : Fin outputCount → Fin inputCount → RegionName × Nat
  write : Fin outputCount → RegionName × Nat
  plans : Fin outputCount → FP32AddPlan inputCount

namespace FP32SumStage

def value {α : Type} [Inhabited α] (stage : FP32SumStage α) (initial : State α)
    (o : Fin stage.outputCount) : α :=
  stage.backend.cast (some .fp32) .real .real
    ((stage.plans o).tree.eval stage.backend
      (fun i => (initial.mem (stage.read o i).1 (stage.read o i).2).read .real))

/-- All input values come from the stage-entry snapshot. Stores follow the
finite increasing output order; later writes win if declared addresses alias. -/
def run {α : Type} [Inhabited α] (stage : FP32SumStage α) (initial : State α) : State α :=
  (List.ofFn (fun o : Fin stage.outputCount => o)).foldl
    (fun current o => current.write (stage.write o).1 (stage.write o).2
      (.mk .real (stage.value initial o))) initial

/-- These are precisely the finite stage's physical memory accesses.
Coverage ensures every input index is an actual tree leaf. -/
def Safe {α : Type} (stage : FP32SumStage α) (bounds : RegionName → Nat) : Prop :=
  (∀ o i, (stage.read o i).2 < bounds (stage.read o i).1) ∧
  (∀ o, (stage.write o).2 < bounds (stage.write o).1)

theorem frame {α : Type} [Inhabited α] (stage : FP32SumStage α) (initial : State α)
    (r : RegionName) (a : Nat)
    (outside : ∀ o, r ≠ (stage.write o).1 ∨ a ≠ (stage.write o).2) :
    (stage.run initial).mem r a = initial.mem r a := by
  have fold : ∀ (xs : List (Fin stage.outputCount)) (s : State α),
      (xs.foldl (fun t o => t.write (stage.write o).1 (stage.write o).2
        (.mk .real (stage.value initial o))) s).mem r a = s.mem r a := by
    intro xs
    induction xs with
    | nil => intro s; rfl
    | cons o rest ih =>
        intro s
        rw [List.foldl_cons, ih, State.write_other _ _ _ _ _ _ (outside o)]
  exact fold _ _

theorem preserves_locals {α : Type} [Inhabited α] (stage : FP32SumStage α) (initial : State α) :
    (stage.run initial).regs = initial.regs ∧
    (stage.run initial).pids = initial.pids ∧
    (stage.run initial).numPids = initial.numPids ∧
    (stage.run initial).undef = initial.undef := by
  have fold : ∀ (xs : List (Fin stage.outputCount)) (s : State α),
      let t := xs.foldl (fun t o => t.write (stage.write o).1 (stage.write o).2
        (.mk .real (stage.value initial o))) s
      t.regs = s.regs ∧ t.pids = s.pids ∧ t.numPids = s.numPids ∧ t.undef = s.undef := by
    intro xs
    induction xs with
    | nil => intro s; exact ⟨rfl, rfl, rfl, rfl⟩
    | cons o rest ih => intro s; exact ih _
  exact fold _ _

end FP32SumStage

/-- Only the two existing concrete execution forms are available. -/
inductive ReductionStage (α : Type) where
  | invocation (job : Invocation α)
  | fp32Sum (stage : FP32SumStage α)

namespace ReductionStage
def run {α : Type} [Inhabited α] : ReductionStage α → State α → Option (State α)
  | .invocation job, s => job.run s
  | .fp32Sum stage, s => some (stage.run s)

def Safe {α : Type} [Inhabited α] (bounds : RegionName → Nat) :
    ReductionStage α → State α → Prop
  | .invocation job, s => job.Safe bounds s
  | .fp32Sum stage, _ => stage.Safe bounds
end ReductionStage

def runReductionStages {α : Type} [Inhabited α] :
    List (ReductionStage α) → State α → Option (State α)
  | [], s => some s
  | stage :: rest, s => (stage.run s).bind (runReductionStages rest)

/-- Every stage observes the actual predecessor, including a kernel that
follows a finite sum and any sum that follows a source write. -/
def ReductionScheduleSafe {α : Type} [Inhabited α] (bounds : RegionName → Nat) :
    List (ReductionStage α) → State α → Prop
  | [], _ => True
  | stage :: rest, s => stage.Safe bounds s ∧
      ∀ t, stage.run s = some t → ReductionScheduleSafe bounds rest t

theorem runReductionStages_append {α : Type} [Inhabited α]
    (a b : List (ReductionStage α)) (s : State α) :
    runReductionStages (a ++ b) s = (runReductionStages a s).bind (runReductionStages b) := by
  induction a generalizing s with
  | nil => rfl
  | cons stage rest ih =>
      simp only [List.cons_append, runReductionStages]
      cases stage.run s <;> simp [ih]

theorem reductionScheduleSafe_append {α : Type} [Inhabited α] (bounds : RegionName → Nat)
    (a b : List (ReductionStage α)) (s : State α) :
    ReductionScheduleSafe bounds (a ++ b) s ↔ ReductionScheduleSafe bounds a s ∧
      ∀ t, runReductionStages a s = some t → ReductionScheduleSafe bounds b t := by
  induction a generalizing s with
  | nil => simp [ReductionScheduleSafe, runReductionStages]
  | cons stage rest ih =>
      simp only [List.cons_append, ReductionScheduleSafe, runReductionStages]
      cases h : stage.run s <;> simp [ih, and_assoc]

theorem runReductionStages_invocations {α : Type} [Inhabited α]
    (jobs : List (Invocation α)) (s : State α) :
    runReductionStages (jobs.map ReductionStage.invocation) s = runSchedule jobs s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons job rest ih =>
      simp only [List.map_cons, runReductionStages, runSchedule, ReductionStage.run]
      cases job.run s <;> simp [ih]

theorem reductionScheduleSafe_invocations {α : Type} [Inhabited α]
    (bounds : RegionName → Nat) (jobs : List (Invocation α)) (s : State α) :
    ReductionScheduleSafe bounds (jobs.map ReductionStage.invocation) s ↔
      ScheduleSafe bounds jobs s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons job rest ih =>
      simp only [List.map_cons, ReductionScheduleSafe, ScheduleSafe,
        ReductionStage.Safe, ReductionStage.run, ih]

structure ReductionProgramIO (α β : Type) extends MemoryIO α where
  stages : ChannelValues α inputs → List (ReductionStage β)
  encode : α → β
  observe : (o : Fin outputs.length) → TileIndex (outputs[o]).shape → β → α

namespace ReductionProgramIO
abbrev Inputs {α β : Type} (io : ReductionProgramIO α β) := io.toMemoryIO.Inputs
abbrev Outputs {α β : Type} (io : ReductionProgramIO α β) := io.toMemoryIO.Outputs
abbrev InBounds {α β : Type} (io : ReductionProgramIO α β) := io.toMemoryIO.InBounds
abbrev InputBinding {α β : Type} [Inhabited β] (io : ReductionProgramIO α β) :=
  io.toMemoryIO.InputBinding io.encode
abbrev OutputBinding {α β : Type} (io : ReductionProgramIO α β) :=
  io.toMemoryIO.OutputBinding io.observe
abbrev Frame {α β : Type} (io : ReductionProgramIO α β) (vals : io.Inputs)
    (before after : State β) := io.toMemoryIO.Frame vals before after

def ImplementsOn {α β : Type} [Inhabited β] (io : ReductionProgramIO α β)
    (domain : io.Inputs → Prop) (f : io.Inputs → io.Outputs) : Prop :=
  ∀ vals, domain vals → ∀ initial, io.InputBinding vals initial →
    (∀ bounds, io.InBounds bounds vals → ReductionScheduleSafe bounds (io.stages vals) initial) ∧
    ∃ final, runReductionStages (io.stages vals) initial = some final ∧
      io.OutputBinding vals (f vals) final ∧ io.Frame vals initial final

def Implements {α β : Type} [Inhabited β] (io : ReductionProgramIO α β)
    (f : io.Inputs → io.Outputs) : Prop := io.ImplementsOn (fun _ => True) f

scoped infix:50 " ⊨ " => Implements
scoped notation:25 io " ⊨{" domain "} " f => ImplementsOn io domain f

theorem ImplementsOn.intro {α β : Type} [Inhabited β] {io : ReductionProgramIO α β}
    {domain : io.Inputs → Prop} {f : io.Inputs → io.Outputs}
    (safe : ∀ vals, domain vals → ∀ bounds, io.InBounds bounds vals → ∀ initial,
      io.InputBinding vals initial → ReductionScheduleSafe bounds (io.stages vals) initial)
    (correct : ∀ vals, domain vals → ∀ initial, io.InputBinding vals initial →
      ∃ final, runReductionStages (io.stages vals) initial = some final ∧
        io.OutputBinding vals (f vals) final ∧ io.Frame vals initial final) :
    io.ImplementsOn domain f := by
  intro vals admitted initial inputs
  exact ⟨fun bounds backed => safe vals admitted bounds backed initial inputs,
    correct vals admitted initial inputs⟩

end ReductionProgramIO
end
end VeriTile.Triton.FP.Structural
