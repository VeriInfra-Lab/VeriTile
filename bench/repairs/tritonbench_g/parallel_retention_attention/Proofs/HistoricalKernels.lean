import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.Contract

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention
noncomputable section

/-- Pinned repaired Python bytes, audited by the retained AST inverse and
eight offline Triton 3.7.1 SM80/SM90 compilations. -/
def historicalSourceSHA256 : String := "678648ade1cedb09c1b0ae40c03155c9ebc09bb812b3cb945a15c98e18ce2ab5"

def forwardGrid (s : Shape) : Nat×Nat×Nat :=
  (blocks s.DK s.BK*blocks s.DV s.BV,blocks s.T 128,s.B*s.H)
def backwardGrid (s : Shape) : Nat×Nat×Nat :=
  (blocks s.DK s.BK*blocks s.DV s.BV,blocks s.T 64,s.B*s.H)

/-- Active stores of the exact full grid. Grid owner and address-injectivity
proofs justify this serialization; every permutation has the same result. -/
def forwardSchedule (s : Shape) := List.range (OutputKind.forward.count s)
def backwardSchedule (s : Shape) := allCells s true
def backwardHostSchedule (s : Shape) := allCells s false

structure ForwardResult (σ : Type) where
  state : σ
  context : SavedContext
  output : TensorHandle

structure BackwardResult (σ : Type) where
  state : σ
  query : TensorHandle
  key : TensorHandle
  value : TensorHandle

/-- Source wrapper execution: full launch into fresh scratch, actual sum(0)
and dtype cast, then save the materialized inputs and return the allocated
output. LocalRuntime specifies only one-cell stores and their observation. -/
def forwardAPI (abi : LocalRuntime α σ) (a : Ops α) (p : Plans) (s : Shape) (b : Bases)
    (tree : SumPlan) (layout : ForwardLayout s b) (saved : SavedInputs s b) (state : σ) : ForwardResult σ :=
  let ret := forwardReturn s saved.query saved.key saved.value layout.output
  ⟨runtimeForward abi a p s b tree layout.scratch layout.output (forwardSchedule s) state,ret.1,ret.2⟩

/-- One shared backward kernel writes dQ,dK,dV; all three host reductions read
that shared completed state. The tuple is in the original API order. -/
def backwardAPI (abi : LocalRuntime α σ) (a : Ops α) (p : Plans) (s : Shape) (b : Bases)
    (trees : OutputKind → SumPlan) (layout : BackwardLayout s b) (context : SavedContext)
    (state : σ) : BackwardResult σ :=
  let ret := backwardReturn s context (layout.output .query) (layout.output .key) (layout.output .value)
  ⟨runtimeBackward abi a p s b trees layout.scratch layout.output
      (backwardSchedule s) (backwardHostSchedule s) state,ret.1,ret.2.1,ret.2.2⟩

def BackwardResult.handle (result : BackwardResult σ) (kind : OutputKind) : TensorHandle :=
  match kind with
  | .query => result.query
  | .key => result.key
  | .value => result.value
  | .forward => result.value

/-- Empty B/H/T is a host no-launch path; every scratch and returned array has
zero active cells, including all three backward results. -/
theorem empty_schedules (s : Shape) (empty : s.B*s.H*s.T=0) :
    forwardSchedule s=[] ∧ backwardSchedule s=[] ∧ backwardHostSchedule s=[] := by
  have z := empty_counts s
  constructor
  · simp [forwardSchedule,(z .forward empty).2]
  · simp [backwardSchedule,backwardHostSchedule,allCells,
      (z .query empty).2,(z .key empty).2,(z .value empty).2,empty]

theorem backward_handle (abi : LocalRuntime α σ) (a : Ops α) (p : Plans) (s : Shape) (b : Bases)
    (trees : OutputKind → SumPlan) (layout : BackwardLayout s b) (context : SavedContext)
    (state : σ) (kind : OutputKind) (active : kind≠.forward) :
    (backwardAPI abi a p s b trees layout context state).handle kind=
      outputHandle s kind (layout.output kind) context.query.device := by
  cases kind <;> simp_all [backwardAPI,BackwardResult.handle,backwardReturn]

#axiomsClean empty_schedules
#axiomsClean backward_handle
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention
