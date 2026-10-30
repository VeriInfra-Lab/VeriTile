import bench.tritonbench_g.int8_matmul_kernel.Whole001.Proofs.Autotune
import bench.tritonbench_g.int8_matmul_kernel.Whole001.Proofs.EventGrid
import bench.tritonbench_g.int8_matmul_kernel.Whole001.Proofs.SourceLink
namespace VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
noncomputable section
set_option maxHeartbeats 4000000

/-- Rank-two, nonnegative-stride tensor descriptor. Storage offsets and inert
singleton/empty strides are retained. Resolved means the logical value equals
the typed raw observation: no lazy negative/conjugate transformation. -/
structure Tensor where
  storage : Nat
  offset : Nat
  rows : Nat
  cols : Nat
  strideRow : Nat
  strideCol : Nat
  dtype : DType
  device : Nat
  cuda : Bool
  resolved : Bool
  capacity : Nat
  deriving DecidableEq

def Tensor.contiguous (t : Tensor) : Prop :=
  t.rows*t.cols=0 ∨ ((t.cols≤1 ∨ t.strideCol=1) ∧ (t.rows≤1 ∨ t.strideRow=max t.cols 1))
instance (t : Tensor) : Decidable t.contiguous := by unfold Tensor.contiguous; infer_instance

def Tensor.covered (t : Tensor) : Prop :=
  ∀ i j,i<t.rows → j<t.cols → t.offset+i*t.strideRow+j*t.strideCol<t.capacity

structure Request where
  a : Tensor
  b : Tensor
  currentDevice : Nat
  deriving DecidableEq

def Request.shape (r : Request) : Shape := ⟨r.a.rows,r.b.cols,r.a.cols⟩
def Request.packedShape (r : Request) : Prop := r.a.cols=4*r.b.rows
instance (r : Request) : Decidable r.packedShape := inferInstanceAs (Decidable (_=_))

def Request.Domain (r : Request) : Prop :=
  r.packedShape ∧ r.a.contiguous ∧ r.a.dtype=.fp32 ∧ r.b.dtype=.uint8 ∧
  r.a.cuda=true ∧ r.b.cuda=true ∧ r.a.device=r.b.device ∧
  r.currentDevice=r.a.device ∧ r.a.resolved=true ∧ r.b.resolved=true ∧
  r.a.covered ∧ r.b.covered

inductive HostDecision where
  | shapeAssertion | contiguityAssertion | allocateAndLaunch
  deriving DecidableEq

/-- Exactly the two source host assertions, in source order. Dtype/device,
resolved-value, bounds, resources, and defined-cast restrictions are public
execution-domain premises; they are not invented source runtime guards. -/
def hostDecision (r : Request) : HostDecision :=
  if ¬r.packedShape then .shapeAssertion
  else if ¬r.a.contiguous then .contiguityAssertion else .allocateAndLaunch

theorem host_shape_failure (r : Request) (h : ¬r.packedShape) : hostDecision r=.shapeAssertion := by
  simp [hostDecision,h]
theorem host_contiguity_failure (r : Request) (shape : r.packedShape) (h : ¬r.a.contiguous) :
    hostDecision r=.contiguityAssertion := by simp [hostDecision,shape,h]
theorem host_success (r : Request) (h : r.Domain) : hostDecision r=.allocateAndLaunch := by
  simp [hostDecision,h.1,h.2.1]

def outputTensor (r : Request) (storage : Nat) : Tensor where
  storage := storage
  offset := 0
  rows := r.a.rows
  cols := r.b.cols
  strideRow := max r.b.cols 1
  strideCol := 1
  dtype := .int32
  device := r.a.device
  cuda := r.a.cuda
  resolved := true
  capacity := r.a.rows*r.b.cols

def hostBuffers (r : Request) (out : Nat) : Buffers :=
  ⟨r.a.storage,r.b.storage,out,r.a.offset,r.b.offset,
    r.a.strideRow,r.a.strideCol,r.b.strideRow,r.b.strideCol⟩

/-- A fresh allocation changes no caller storage. It may initialize its own
new storage arbitrarily; all logical C cells are subsequently overwritten. -/
structure Allocation (r : Request) (caller : Set Nat) (before : Memory α) where
  output : Nat
  fresh : output∉caller
  aCaller : r.a.storage∈caller
  bCaller : r.b.storage∈caller
  initial : Memory α
  preserves : ∀ p,p.1.1≠output → initial p=before p

variable {r : Request} {caller : Set Nat} {before : Memory α}

def Allocation.buffers (h : Allocation r caller before) := hostBuffers r h.output
def Allocation.returned (h : Allocation r caller before) := outputTensor r h.output

theorem Allocation.separate (h : Allocation r caller before) : Separate h.buffers := by
  change h.output≠r.a.storage ∧ h.output≠r.b.storage
  constructor
  · intro he; exact h.fresh (by rw [he]; exact h.aCaller)
  · intro he; exact h.fresh (by rw [he]; exact h.bCaller)

theorem Allocation.inputs (h : Allocation r caller before) (p : Address) (read : h.buffers.reads p) :
    h.initial p=before p := by
  apply h.preserves
  intro he
  rcases read with ha|hb
  · have hs := congrArg Prod.fst ha
    exact h.separate.1 (he.symm.trans hs)
  · have hs := congrArg Prod.fst hb
    exact h.separate.2 (he.symm.trans hs)

/-- Source always submits this grid, including zero. `run_empty` describes
runtime no-op semantics of a zero grid, not a host-side skip. -/
def launchRequest (r : Request) (c : Fin 15) : Nat := r.shape.programs c

def Allocation.finalMemory (h : Allocation r caller before) (a : Arithmetic α β)
    (trials : List (Fin 15)) (selected : Fin 15) : Memory α :=
  run a r.shape h.buffers selected (trialRun a r.shape h.buffers trials h.initial)

theorem Allocation.final_selected (h : Allocation r caller before) (a : Arithmetic α β)
    (trials : List (Fin 15)) (selected : Fin 15) (legal : r.shape.legal selected)
    (trialLegal : ∀ c∈trials,r.shape.legal c) :
    h.finalMemory a trials selected=run a r.shape h.buffers selected h.initial :=
  trial_absorbed a _ _ trials selected legal trialLegal h.separate h.initial

theorem Allocation.output_values (h : Allocation r caller before) (a : Arithmetic α β)
    (trials : List (Fin 15)) (selected : Fin 15) (legal : r.shape.legal selected)
    (trialLegal : ∀ c∈trials,r.shape.legal c) (m n : Nat) (hm : m<r.shape.m) (hn : n<r.shape.n) :
    h.finalMemory a trials selected (outAddress r.shape h.buffers m n)=
      a.encode (callerAccumulation a r.shape h.buffers selected before m n) := by
  rw [h.final_selected a trials selected legal trialLegal,run_output a _ _ _ legal h.separate _ _ _ hm hn,
    accumulation_reads a _ _ _ before h.initial h.inputs]

theorem Allocation.caller_frame (h : Allocation r caller before) (a : Arithmetic α β)
    (trials : List (Fin 15)) (selected : Fin 15) (legal : r.shape.legal selected)
    (trialLegal : ∀ c∈trials,r.shape.legal c) (p : Address) (owned : p.1.1∈caller) :
    h.finalMemory a trials selected p=before p := by
  have neq : p.1.1≠h.output := by intro he; exact h.fresh (he ▸ owned)
  rw [h.final_selected a trials selected legal trialLegal,
    run_frame a _ _ _ legal h.separate h.initial p (by
      intro write; exact neq (congrArg Prod.fst write.1)),h.preserves p neq]

def sequenceItems (f : δ → Code α β Unit) : List δ → Code α β Unit
  | [] => pure ()
  | q::tail => do
      let _ ← f q
      sequenceItems f tail

theorem eval_sequenceItems (a : Arithmetic α β) (f : δ → Code α β Unit)
    (step : δ → Memory α → Memory α) (tasks : List δ)
    (localRule : ∀ q∈tasks,∀ m,(f q).eval a m=((),step q m)) (m : Memory α) :
    (sequenceItems f tasks).eval a m=((),Schedule.run step tasks m) := by
  induction tasks generalizing m with
  | nil => rfl
  | cons q tail ih =>
    simp only [sequenceItems,bind,eval_bind,localRule q (by simp),Prod.fst,Prod.snd]
    rw [ih (by intro x hx; exact localRule x (List.mem_cons_of_mem q hx))]
    rfl

def Allocation.launchCode (h : Allocation r caller before) (trials : List (Fin 15)) (c : Fin 15) : Code α β Unit := do
  let _ ← sequenceItems (gridCode r.shape h.buffers) trials
  gridCode r.shape h.buffers c

theorem Allocation.eval_launchCode (h : Allocation r caller before) (a : Arithmetic α β)
    (trials : List (Fin 15)) (c : Fin 15) (legal : r.shape.legal c)
    (trialLegal : ∀ c∈trials,r.shape.legal c) :
    (h.launchCode trials c).eval a h.initial=((),h.finalMemory a trials c) := by
  simp only [Allocation.launchCode,bind,eval_bind]
  rw [eval_sequenceItems a _ _ trials (fun q hq m => eval_gridCode a _ _ q m (trialLegal q hq) h.separate)]
  simp only [Prod.fst,Prod.snd]
  exact eval_gridCode a _ _ c _ legal h.separate

theorem Allocation.native_launch (h : Allocation r caller before) (a : Arithmetic α β)
    (trials : List (Fin 15)) (c : Fin 15) (legal : r.shape.legal c)
    (trialLegal : ∀ c∈trials,r.shape.legal c) (abi : LocalEventABI α β σ a)
    (state : σ) (entry : abi.observe state=h.initial) :
    abi.observe ((h.launchCode trials c).native abi state).2=h.finalMemory a trials c := by
  have proof := Code.native_refines abi (h.launchCode trials c) state
  rw [entry,h.eval_launchCode a trials c legal trialLegal] at proof
  exact congrArg Prod.snd proof

#axiomsClean host_success
#axiomsClean Allocation.inputs
#axiomsClean Allocation.output_values
#axiomsClean Allocation.caller_frame
#axiomsClean Allocation.native_launch
end
end VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
