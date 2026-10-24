import VeriTile.Triton.Memory.RegionalKernelIO

/-! Finite whole-source scheduling for the existing Algorithm regional IO.

The channel types, windows, masks, source projection and exact raw-cell frame
are those of `UKernelIO`. Every launch executes that same projected source;
launch data can install program IDs/grid dimensions but cannot supply a runner
or alter memory/registers. Success and outputs do not require a finite capacity
witness. Capacity bounds separately imply safety at every actual predecessor.
-/
namespace VeriTile.Triton
noncomputable section

/-- Data-only launch metadata. Unselected axes retain their current values. -/
structure AlgorithmLaunch where
  pids : Nat → Nat
  setPids : Nat → Bool := fun _ => true
  numPids : Option (Nat → Nat) := none

namespace AlgorithmLaunch

def initial (launch : AlgorithmLaunch) (s : BlockState) : BlockState :=
  { s with
    pids := fun ax => if launch.setPids ax then launch.pids ax else s.pids ax
    numPids := launch.numPids.getD s.numPids }

@[simp] theorem initial_mem (launch : AlgorithmLaunch) (s : BlockState) :
    (launch.initial s).mem = s.mem := rfl

@[simp] theorem initial_regs (launch : AlgorithmLaunch) (s : BlockState) :
    (launch.initial s).regs = s.regs := rfl

@[simp] theorem initial_undef (launch : AlgorithmLaunch) (s : BlockState) :
    (launch.initial s).undef = s.undef := rfl

def run (launch : AlgorithmLaunch) (kernel : Kernel) (s : BlockState) :
    Option BlockState := exec kernel (launch.initial s)

/-- Identity-rounding source safety follows the same exact execution:
`execR_triv` and `stepStmtR_triv` identify its predecessor semantics. -/
def Safe (launch : AlgorithmLaunch) (kernel : Kernel) (bounds : RegionBounds)
    (s : BlockState) : Prop :=
  Kernel.TraceSafeR .triv bounds kernel (launch.initial s)

end AlgorithmLaunch

def runAlgorithmSchedule (kernel : Kernel) : List AlgorithmLaunch → BlockState → Option BlockState
  | [], s => some s
  | launch :: rest, s => (launch.run kernel s).bind (runAlgorithmSchedule kernel rest)

/-- The next launch is checked in the actual successful predecessor state. -/
def AlgorithmScheduleSafe (kernel : Kernel) (bounds : RegionBounds) :
    List AlgorithmLaunch → BlockState → Prop
  | [], _ => True
  | launch :: rest, s => launch.Safe kernel bounds s ∧
      ∀ t, launch.run kernel s = some t → AlgorithmScheduleSafe kernel bounds rest t

@[simp] theorem runAlgorithmSchedule_singleton (kernel : Kernel)
    (launch : AlgorithmLaunch) (s : BlockState) :
    runAlgorithmSchedule kernel [launch] s = launch.run kernel s := by
  cases ran : launch.run kernel s <;> simp [runAlgorithmSchedule, ran]

@[simp] theorem algorithmScheduleSafe_singleton (kernel : Kernel) (bounds : RegionBounds)
    (launch : AlgorithmLaunch) (s : BlockState) :
    AlgorithmScheduleSafe kernel bounds [launch] s ↔ launch.Safe kernel bounds s := by
  simp [AlgorithmScheduleSafe]

theorem runAlgorithmSchedule_append (kernel : Kernel)
    (left right : List AlgorithmLaunch) (s : BlockState) :
    runAlgorithmSchedule kernel (left ++ right) s =
      (runAlgorithmSchedule kernel left s).bind (runAlgorithmSchedule kernel right) := by
  induction left generalizing s with
  | nil => rfl
  | cons launch rest ih =>
    simp only [List.cons_append, runAlgorithmSchedule]
    cases ran : launch.run kernel s <;> simp [ih]

theorem algorithmScheduleSafe_append (kernel : Kernel) (bounds : RegionBounds)
    (left right : List AlgorithmLaunch) (s : BlockState) :
    AlgorithmScheduleSafe kernel bounds (left ++ right) s ↔
      AlgorithmScheduleSafe kernel bounds left s ∧
      ∀ t, runAlgorithmSchedule kernel left s = some t →
        AlgorithmScheduleSafe kernel bounds right t := by
  induction left generalizing s with
  | nil => simp [AlgorithmScheduleSafe, runAlgorithmSchedule]
  | cons launch rest ih =>
    simp only [List.cons_append, AlgorithmScheduleSafe, runAlgorithmSchedule]
    cases ran : launch.run kernel s <;> simp [ih, and_assoc]

/-- Convenient when source safety is already known at every possible state. -/
theorem algorithmScheduleSafe_of_forall (kernel : Kernel) (bounds : RegionBounds)
    (jobs : List AlgorithmLaunch)
    (safe : ∀ launch ∈ jobs, ∀ s, launch.Safe kernel bounds s) (s : BlockState) :
    AlgorithmScheduleSafe kernel bounds jobs s := by
  induction jobs generalizing s with
  | nil => trivial
  | cons launch rest ih =>
    refine ⟨safe launch (by simp) s, ?_⟩
    intro t _
    exact ih (fun other member => safe other (by simp [member])) t

namespace UKernelIO

/-- Complete scheduled regional contract. Windows describe the whole launch
family, using initial metadata; each source execution uses its installed IDs.
The source projection obligation is inherited unchanged from `UKernelIO`.
-/
def RegionalScheduledImplements3D (io : UKernelIO)
    (schedule : io.Ctx → List AlgorithmLaunch)
    (f : Nat → Nat → Nat → io.Ctx → (o : Fin io.nOut) →
      Fin (io.oarity o) → (io.oty o).carrier) : Prop :=
  ∀ (s : BlockState) (vals : io.Ctx), io.RegionalInputs s vals →
    (∀ bounds, io.RegionalBounds bounds s vals →
      AlgorithmScheduleSafe io.kernel.toAlgKernel bounds (schedule vals) s) ∧
    ∃ final, runAlgorithmSchedule io.kernel.toAlgKernel (schedule vals) s = some final ∧
      (∀ (o : Fin io.nOut) (j : Fin (io.oarity o)),
        io.omask o vals (s.pids 0) (s.pids 1) (s.pids 2) j →
        (io.oty o).read final (io.obuf o)
          (io.owin o vals (s.pids 0) (s.pids 1) (s.pids 2) j) =
          f (s.pids 0) (s.pids 1) (s.pids 2) vals o j) ∧
      io.RegionalFrame s final vals

/-- Whole-grid formulas that do not depend on the initial program IDs. -/
def RegionalScheduledImplements (io : UKernelIO)
    (schedule : io.Ctx → List AlgorithmLaunch)
    (f : io.Ctx → (o : Fin io.nOut) → Fin (io.oarity o) → (io.oty o).carrier) : Prop :=
  io.RegionalScheduledImplements3D schedule (fun _ _ _ => f)

theorem RegionalScheduledImplements3D.intro (io : UKernelIO)
    {schedule : io.Ctx → List AlgorithmLaunch}
    {f : Nat → Nat → Nat → io.Ctx → (o : Fin io.nOut) →
      Fin (io.oarity o) → (io.oty o).carrier}
    (safe : ∀ s vals, io.RegionalInputs s vals → ∀ bounds,
      io.RegionalBounds bounds s vals →
      AlgorithmScheduleSafe io.kernel.toAlgKernel bounds (schedule vals) s)
    (correct : ∀ s vals, io.RegionalInputs s vals →
      ∃ final, runAlgorithmSchedule io.kernel.toAlgKernel (schedule vals) s = some final ∧
        (∀ (o : Fin io.nOut) (j : Fin (io.oarity o)),
          io.omask o vals (s.pids 0) (s.pids 1) (s.pids 2) j →
          (io.oty o).read final (io.obuf o)
            (io.owin o vals (s.pids 0) (s.pids 1) (s.pids 2) j) =
            f (s.pids 0) (s.pids 1) (s.pids 2) vals o j) ∧
        io.RegionalFrame s final vals) :
    io.RegionalScheduledImplements3D schedule f := by
  intro s vals inputs
  exact ⟨safe s vals inputs, correct s vals inputs⟩

theorem RegionalScheduledImplements.intro (io : UKernelIO)
    {schedule : io.Ctx → List AlgorithmLaunch}
    {f : io.Ctx → (o : Fin io.nOut) → Fin (io.oarity o) → (io.oty o).carrier}
    (safe : ∀ s vals, io.RegionalInputs s vals → ∀ bounds,
      io.RegionalBounds bounds s vals →
      AlgorithmScheduleSafe io.kernel.toAlgKernel bounds (schedule vals) s)
    (correct : ∀ s vals, io.RegionalInputs s vals →
      ∃ final, runAlgorithmSchedule io.kernel.toAlgKernel (schedule vals) s = some final ∧
        (∀ (o : Fin io.nOut) (j : Fin (io.oarity o)),
          io.omask o vals (s.pids 0) (s.pids 1) (s.pids 2) j →
          (io.oty o).read final (io.obuf o)
            (io.owin o vals (s.pids 0) (s.pids 1) (s.pids 2) j) = f vals o j) ∧
        io.RegionalFrame s final vals) :
    io.RegionalScheduledImplements schedule f :=
  RegionalScheduledImplements3D.intro io safe correct

end UKernelIO
end
end VeriTile.Triton
