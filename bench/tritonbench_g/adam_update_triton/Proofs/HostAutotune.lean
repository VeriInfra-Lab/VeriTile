import bench.tritonbench_g.adam_update_triton.Proofs.GridFrameReuse

/-! Host/autotune composition for the source-selected FP32 Adam/Lion path.
Torch metadata and the autotuner are runtime contracts. In particular, the
runtime never supplies the final output: the final selected launch is the
original kernel's existing GridLaunchedOrdinary execution.

This is a mathematical region-memory helper. Tensor-to-region/stride binding,
native FP32 values and scalar ABI, native indices/addresses and unrestricted
aliases remain separate obligations. The source has no shape-equality test.
-/

namespace VeriTile.Bench.TritonBenchG.AdamUpdateTriton.HostAutotune
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

/-- Exactly the source autotune alternatives, including num_warps. -/
inductive Config where
  | block128warps4 | block1024warps8
  deriving DecidableEq, Repr

def Config.block : Config → Nat
  | .block128warps4 => 128
  | .block1024warps8 => 1024

def Config.warps : Config → Nat
  | .block128warps4 => 4
  | .block1024warps8 => 8

def sourceConfigs : List Config := [.block128warps4, .block1024warps8]

theorem config_source (c : Config) :
    (c.block = 128 ∧ c.warps = 4) ∨ (c.block = 1024 ∧ c.warps = 8) := by
  cases c <;> simp [Config.block, Config.warps]

theorem config_registered (c : Config) : c ∈ sourceConfigs := by
  cases c <;> simp [sourceConfigs]

/-- Ordinary tensor metadata contract. This helper fixes the source-selected
FP32 route; shape is not asserted equal across the three arguments. The region
is the separately justified linear storage interpretation of that tensor. -/
structure Tensor where
  region : RegionName
  shape : List Nat
  isCuda : Bool

def Tensor.numel (t : Tensor) : Nat := t.shape.prod

structure Arguments where
  p : Tensor
  grad : Tensor
  expAvg : Tensor
  lr : ℝ
  wd : ℝ
  beta1 : ℝ
  beta2 : ℝ

def allCuda (a : Arguments) : Bool := a.p.isCuda && a.grad.isCuda && a.expAvg.isCuda

structure Request where
  arguments : Arguments
  nElements : Nat

def request (a : Arguments) : Request := ⟨a, a.p.numel⟩

def Request.grid (r : Request) (c : Config) : Nat :=
  (r.nElements + c.block - 1) / c.block

def Request.key (r : Request) : List Nat := [r.nElements]

def Request.restorePointers (r : Request) : List RegionName :=
  [r.arguments.p.region, r.arguments.expAvg.region]

/-- The CUDA assertion precedes p.numel and the runtime call. No shape,
same-device, contiguity or nonempty assertion is added to the Python source. -/
def prepare (a : Arguments) : Except Unit Request :=
  if allCuda a then .ok (request a) else .error ()

theorem prepare_cuda (a : Arguments) (h : allCuda a = Bool.true) :
    prepare a = .ok (request a) := by simp [prepare, h]

theorem prepare_assertion (a : Arguments) (h : allCuda a = Bool.false) :
    prepare a = .error () := by simp [prepare, h]

theorem source_request (a : Arguments) :
    (request a).nElements = a.p.shape.prod ∧
    (request a).key = [a.p.shape.prod] ∧
    (request a).restorePointers = [a.p.region, a.expAvg.region] := by
  exact ⟨rfl, rfl, rfl⟩

theorem empty_grid (a : Arguments) (h : a.p.numel = 0) (c : Config) :
    (request a).grid c = 0 := by
  cases c <;> simp [Request.grid, request, h, Config.block]

/-- Logical cells restored by the two source restore_value tensor snapshots.
The momentum snapshot follows its own numel, not an invented shape assertion. -/
def RestoreFootprint (r : Request) (region : RegionName) (offset : Nat) : Prop :=
  (region = r.arguments.p.region ∧ offset < r.arguments.p.numel) ∨
  (region = r.arguments.expAvg.region ∧ offset < r.arguments.expAvg.numel)

def TrialWrites (r : Request) (region : RegionName) (offset : Nat) : Prop :=
  (region = r.arguments.p.region ∨ region = r.arguments.expAvg.region) ∧
    offset < r.nElements

/-- Runtime clone/copy restoration contract. It refers only to entry snapshots
and the trial state; it contains no Lion formula or final kernel output. -/
structure RestoreContract (r : Request) (initial trial restored : BlockState) : Prop where
  snapshot : ∀ region offset, RestoreFootprint r region offset →
    restored.mem region offset = initial.mem region offset
  frame : ∀ region offset, ¬ RestoreFootprint r region offset →
    restored.mem region offset = trial.mem region offset

/-- A finite successful autotuning/cache result. Repeated trials and a cache
hit (empty list) are permitted. Trial effects are arbitrary inside the source
write footprint. Runtime failures are retained by RuntimeReply below.
Selection timing, Torch clone/copy and runtime internals are trusted contracts;
none of them returns the selected kernel's final result. -/
structure TuneReceipt (r : Request) (initial : BlockState) where
  trials : List Config
  selected : Config
  afterTrials : BlockState
  launchInitial : BlockState
  trialFrame : ∀ region offset, ¬ TrialWrites r region offset →
    afterTrials.mem region offset = initial.mem region offset
  restore : RestoreContract r initial afterTrials launchInitial

inductive RuntimeReply (r : Request) (initial : BlockState) where
  | failed (reason : String) (state : BlockState)
  | ready (receipt : TuneReceipt r initial)

abbrev Runtime := (r : Request) → (initial : BlockState) → RuntimeReply r initial

/-- Every kernel-written cell must lie in a restored tensor snapshot. This is
a theorem applicability condition, not a new Python assertion. In particular,
exp_avg with fewer elements than p is not silently claimed safe for autotune. -/
theorem restored_memory (r : Request) (initial : BlockState)
    (receipt : TuneReceipt r initial)
    (pCovers : r.nElements ≤ r.arguments.p.numel)
    (momentumCovers : r.nElements ≤ r.arguments.expAvg.numel) :
    ∀ region offset, receipt.launchInitial.mem region offset = initial.mem region offset := by
  intro region offset
  by_cases hr : RestoreFootprint r region offset
  · exact receipt.restore.snapshot region offset hr
  · rw [receipt.restore.frame region offset hr]
    apply receipt.trialFrame region offset
    rintro ⟨hp | hm, active⟩
    · exact hr (Or.inl ⟨hp, lt_of_lt_of_le active pCovers⟩)
    · exact hr (Or.inr ⟨hm, lt_of_lt_of_le active momentumCovers⟩)

inductive Outcome where
  | cudaAssertion (state : BlockState)
  | runtimeFailure (reason : String) (state : BlockState)
  | returnedNone (state : BlockState)

/-- Ordered source wrapper execution. Only a successful prepared request may
call the runtime; only a ready runtime result may run the selected kernel.
The Python function mutates the two inputs and returns None. -/
inductive Executes (runtime : Runtime) (a : Arguments) (initial : BlockState) : Outcome → Prop
  | assertion (failed : prepare a = .error ()) :
      Executes runtime a initial (.cudaAssertion initial)
  | runtimeFailure (prepared : prepare a = .ok (request a))
      (reason : String) (state : BlockState)
      (failed : runtime (request a) initial = .failed reason state) :
      Executes runtime a initial (.runtimeFailure reason state)
  | returned (prepared : prepare a = .ok (request a))
      (receipt : TuneReceipt (request a) initial)
      (ready : runtime (request a) initial = .ready receipt)
      (final : BlockState)
      (launched : Nonempty (Kernel.GridLaunchedOrdinary
        (update_fn_kernel a.p.region a.grad.region a.expAvg.region
          a.lr a.wd a.beta1 a.beta2 a.p.numel receipt.selected.block).toAlgKernel
        (VeriTile.Bench.Examples.AdamUpdateGridLaunch.adamGrid
          ((request a).grid receipt.selected)) receipt.launchInitial final)) :
      Executes runtime a initial (.returnedNone final)

theorem cuda_failure_before_runtime (runtime : Runtime) (a : Arguments)
    (initial : BlockState) (notCuda : allCuda a = Bool.false) :
    Executes runtime a initial (.cudaAssertion initial) :=
  .assertion (prepare_assertion a notCuda)

theorem runtime_failure_preserved (runtime : Runtime) (a : Arguments)
    (initial : BlockState) (cuda : allCuda a = Bool.true)
    (reason : String) (state : BlockState)
    (failed : runtime (request a) initial = .failed reason state) :
    Executes runtime a initial (.runtimeFailure reason state) :=
  .runtimeFailure (prepare_cuda a cuda) reason state failed

/-- One real selected launch after restore, reused without rebuilding its
kernel proof. Both output formulas and the exact cell frame refer to entry
memory, independently of arbitrary trial writes. No final-result callback is
present in the runtime interface. -/
theorem update_fn_success (runtime : Runtime) (a : Arguments) (initial : BlockState)
    (cuda : allCuda a = Bool.true) (distinct : a.p.region ≠ a.expAvg.region)
    (momentumCovers : a.p.numel ≤ a.expAvg.numel)
    (receipt : TuneReceipt (request a) initial)
    (ready : runtime (request a) initial = .ready receipt) :
    ∃ final : BlockState,
      Executes runtime a initial (.returnedNone final) ∧
      (∀ k, k < a.p.numel →
        final.readMem a.p.region k =
          TiledOptimizer.lionParam (initial.readMem a.p.region k)
            (initial.readMem a.expAvg.region k) (initial.readMem a.grad.region k)
            a.lr a.wd a.beta1 ∧
        final.readMem a.expAvg.region k =
          TiledOptimizer.lionMomentum (initial.readMem a.expAvg.region k)
            (initial.readMem a.grad.region k) a.beta2) ∧
      (∀ region offset, (region ≠ a.p.region ∧ region ≠ a.expAvg.region) ∨ a.p.numel ≤ offset →
        final.mem region offset = initial.mem region offset) := by
  have hm := restored_memory (request a) initial receipt (Nat.le_refl _) momentumCovers
  have reads : ∀ region offset,
      receipt.launchInitial.readMem region offset = initial.readMem region offset := by
    intro region offset
    simp [BlockState.readMem, hm]
  have selected : receipt.selected.block = 128 ∨ receipt.selected.block = 1024 := by
    rcases config_source receipt.selected with h | h
    · exact Or.inl h.1
    · exact Or.inr h.1
  obtain ⟨final, launched, values, frame⟩ :=
    GridReuse.selected_config_real_grid_with_frame a.p.region a.grad.region a.expAvg.region
      a.lr a.wd a.beta1 a.beta2 a.p.numel receipt.selected.block selected distinct receipt.launchInitial
  refine ⟨final, .returned (prepare_cuda a cuda) receipt ready final launched, ?_, ?_⟩
  · intro k hk
    simpa only [reads] using values k hk
  · intro region offset outside
    exact (frame region offset outside).trans (hm region offset)

#axiomsClean config_source
#axiomsClean config_registered
#axiomsClean prepare_cuda
#axiomsClean prepare_assertion
#axiomsClean source_request
#axiomsClean empty_grid
#axiomsClean restored_memory
#axiomsClean cuda_failure_before_runtime
#axiomsClean runtime_failure_preserved
#axiomsClean update_fn_success
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.AdamUpdateTriton.HostAutotune
