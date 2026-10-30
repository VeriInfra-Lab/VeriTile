import bench.tritonbench_g.parallel_retention_attention.Proofs.BackwardHost

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention
noncomputable section

/-- Ordinary one-cell memory-store contract. This contains no arithmetic,
whole output, reduction, launch-success, or whole-frame callback. -/
structure LocalRuntime (α σ : Type) where
  observe : σ → Memory α
  write : σ → Nat → α → σ
  write_at : ∀ state address value query,
    observe (write state address value) query=
      if query=address then value else observe state query

def runtimeWrites {ι : Type} (abi : LocalRuntime α σ) (addr : ι → Nat)
    (value : Memory α → ι → α) : List ι → σ → σ
  | [],state => state
  | i::is,state => runtimeWrites abi addr value is (abi.write state (addr i) (value (abi.observe state) i))

theorem runtimeWrites_observe {ι : Type} (abi : LocalRuntime α σ) (addr : ι → Nat)
    (value : Memory α → ι → α) (schedule : List ι) (state : σ) :
    abi.observe (runtimeWrites abi addr value schedule state)=
      runWrites addr value schedule (abi.observe state) := by
  induction schedule generalizing state with
  | nil => rfl
  | cons i is ih =>
    simp only [runtimeWrites,runWrites]
    rw [ih]
    congr 1
    funext query
    rw [abi.write_at]
    simp [Function.update,eq_comm]

theorem runWrites_eq_runStores (addr : Nat → Nat) (value : Memory α → Nat → α)
    (schedule : List Nat) (m : Memory α) : runWrites addr value schedule m=runStores addr value schedule m := by
  induction schedule generalizing m with
  | nil => rfl
  | cons i is ih => exact ih _

def runtimeForward (abi : LocalRuntime α σ) (a : Ops α) (p : Plans) (s : Shape) (b : Bases)
    (tree : SumPlan) (scratch output : Nat) (schedule : List Nat) (state : σ) : σ :=
  let launchState := runtimeWrites abi (fun i => scratch+i) (scratchCurrent a p s b .forward) schedule state
  runtimeWrites abi (fun i => output+i)
    (fun mem i => hostValue a tree (OutputKind.forward.shards s) scratch (OutputKind.forward.cells s) mem i)
    (List.range (OutputKind.forward.cells s)) launchState

theorem runtimeForward_observe (abi : LocalRuntime α σ) (a : Ops α) (p : Plans) (s : Shape) (b : Bases)
    (tree : SumPlan) (scratch output : Nat) (schedule : List Nat) (state : σ) :
    abi.observe (runtimeForward abi a p s b tree scratch output schedule state)=
      wholeRun a p s b .forward tree scratch output schedule (abi.observe state) := by
  unfold runtimeForward
  rw [runtimeWrites_observe,runtimeWrites_observe,runWrites_eq_runStores,runWrites_eq_runStores]
  rfl

def runtimeBackward (abi : LocalRuntime α σ) (a : Ops α) (p : Plans) (s : Shape) (b : Bases)
    (trees : OutputKind → SumPlan) (scratchBases outputBases : OutputKind → Nat)
    (scratchSchedule hostSchedule : List Cell) (state : σ) : σ :=
  let launchState := runtimeWrites abi (cellsAddress scratchBases)
    (fun mem cell => scratchCurrent a p s b cell.kind mem cell.index) scratchSchedule state
  runtimeWrites abi (cellsAddress outputBases) (backwardHostValue a s trees scratchBases) hostSchedule launchState

theorem runtimeBackward_observe (abi : LocalRuntime α σ) (a : Ops α) (p : Plans) (s : Shape) (b : Bases)
    (trees : OutputKind → SumPlan) (scratchBases outputBases : OutputKind → Nat)
    (scratchSchedule hostSchedule : List Cell) (state : σ) :
    abi.observe (runtimeBackward abi a p s b trees scratchBases outputBases scratchSchedule hostSchedule state)=
      backwardWhole a p s b trees scratchBases outputBases scratchSchedule hostSchedule (abi.observe state) := by
  unfold runtimeBackward
  rw [runtimeWrites_observe,runtimeWrites_observe]
  rfl

/-- Concrete inhabitant of the ordinary local-memory interface. -/
def pureRuntime (α : Type) : LocalRuntime α (Memory α) where
  observe := id
  write := Function.update
  write_at := by intros; simp [Function.update,eq_comm]

structure DeviceState (σ : Type) where
  state : σ
  device : Nat

def owningDevice (owner : Nat) (run : σ → σ) (s : DeviceState σ) : DeviceState σ :=
  let entered : DeviceState σ := ⟨s.state,owner⟩
  let finished := run entered.state
  ⟨finished,s.device⟩

theorem device_restored (owner : Nat) (run : σ → σ) (s : DeviceState σ) :
    (owningDevice owner run s).state=run s.state ∧ (owningDevice owner run s).device=s.device := ⟨rfl,rfl⟩

#axiomsClean runtimeWrites_observe
#axiomsClean runtimeForward_observe
#axiomsClean runtimeBackward_observe
#axiomsClean pureRuntime
#axiomsClean device_restored
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention
