import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.Source
import Mathlib.Data.Nat.Log
import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.Contract

namespace VeriTile.Bench.Repairs.L2NormTriton1.Host
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open Physical
noncomputable section
open Classical
set_option maxHeartbeats 5000000

/-- Observation after the selected current stream's preceding work completes.
This does not assert that the Python return synchronizes the device. -/
structure World (M : Type) where
  memory : M
  allocations : List Allocation

inductive Result (M α : Type) where
  | ok (value : α) (state : World M)
  | error (reason : String) (state : World M)

def Result.state {M α : Type} : Result M α → World M
  | .ok _ s => s | .error _ s => s

def Result.bind {M α β : Type} (r : Result M α) (next : α → World M → Result M β) : Result M β :=
  match r with | .ok value s => next value s | .error reason s => .error reason s

/-- Every existing byte and every backed word, not merely the selected tensor
coordinates, is preserved. Allocation bookkeeping may add fresh allocations. -/
structure Keep {M α β : Type} (ops : Ops M α β) (before after : World M) : Prop where
  live : ∀ a∈before.allocations,a∈after.allocations
  reads : ∀ a∈before.allocations,∀ key,a.contains key → ops.read after.memory key=ops.read before.memory key
  bytes : ∀ a∈before.allocations,∀ owner address,a.covers owner address →
    ops.observe after.memory owner address=ops.observe before.memory owner address

theorem Keep.refl {M α β : Type} (ops : Ops M α β) (s : World M) : Keep ops s s :=
  ⟨fun _ h => h,fun _ _ _ _ => rfl,fun _ _ _ _ _ => rfl⟩

theorem Keep.trans {M α β : Type} (ops : Ops M α β) {s t u : World M}
    (h : Keep ops s t) (k : Keep ops t u) : Keep ops s u :=
  ⟨fun a ha => k.live a (h.live a ha),
    fun a ha key hk => (k.reads a (h.live a ha) key hk).trans (h.reads a ha key hk),
    fun a ha owner address hk =>
      (k.bytes a (h.live a ha) owner address hk).trans (h.bytes a ha owner address hk)⟩

/-- Ordinary dense FP32 tensor metadata, including the actual returned object
and allocation. Singleton dimensions may retain irrelevant stride choices;
the logical contiguous coordinates are what the kernel consumes. -/
structure Dense where
  view : View
  rows : Nat
  cols : Nat
  shape : view.shape=[rows,cols]
  normal : view.negative=Bool.false
  dense : ∀ i,i<rows*cols → view.key i=view.storage.key i
  capacity : rows*cols≤view.storage.words

def Dense.count (d : Dense) : Nat := d.rows*d.cols

theorem Dense.size (d : Dense) : d.view.numel=d.count := by simp [View.numel,d.shape,Dense.count]
theorem Dense.backed (d : Dense) : d.view.backed := by
  intro i hi
  rw [d.size] at hi
  rw [d.dense i hi]
  exact Allocation.key_contains _ _ (lt_of_lt_of_le hi d.capacity)

def blockSize (N : Nat) : Nat := min 16384 (2^Nat.clog 2 N)

theorem block_limit (N : Nat) : blockSize N≤16384 := Nat.min_le_left _ _

theorem block_covers (N : Nat) (accepted : N≤16384) : N≤blockSize N :=
  Nat.le_min.mpr ⟨accepted,Nat.le_pow_clog (by decide) N⟩

theorem block_power (N : Nat) (accepted : N≤16384) :
    blockSize N=2^Nat.clog 2 N ∧ Nat.clog 2 N≤14 := by
  have bound : Nat.clog 2 N≤14 := Nat.clog_le_of_le_pow (by simpa using accepted)
  have power : 2^Nat.clog 2 N≤16384 := by
    exact le_trans (Nat.pow_le_pow_right (by decide) bound) (by norm_num)
  exact ⟨Nat.min_eq_right power,bound⟩

inductive Launch (p : Layout) where
  | complete (schedule : List Nat) (allRows : schedule.Perm (List.range p.rows))
  | interrupted (completed : List Nat) (storesDone : List (Nat × Nat))
      (completedBounded : ∀ row∈completed,row<p.rows)
      (storesBounded : ∀ ij∈storesDone,ij.1<p.rows ∧ ij.2<p.cols)
      (reason : String)

structure Runtime (M : Type) where
  enter : Nat → World M → Result M Nat
  leave : Nat → World M → Result M Unit
  normalize : View → World M → Result M Dense
  empty : Nat → Nat → Nat → World M → Result M Dense
  setup : (p : Layout) → World M → Result M (Launch p)
  reshape : Dense → List Nat → Option View

structure Call (α : Type) where
  input : View
  eps : α
  device : Nat
  selectedFP32 : Bool

structure Domain {M α : Type} (call : Call α) (s : World M) : Prop where
  backed : call.input.backed
  live : call.input.storage∈s.allocations
  device : call.device=call.input.storage.owner
  direct : DirectInput call.input

structure Snapshot {M α β : Type} (ops : Ops M α β) (neg : Negation α)
    (input : View) (before : World M) (d : Dense) (after : World M) : Prop where
  keep : Keep ops before after
  live : d.view.storage∈after.allocations
  storageObject : d.view.storage.object=input.storage.object
  device : d.view.storage.owner=input.storage.owner
  positive : 0<d.cols
  cols : d.cols=input.shape.getLastD 0
  count : d.count=input.numel
  keys : ∀ i,i<d.count → d.view.storage.key i=input.key i

structure Allocated {M α β : Type} (ops : Ops M α β)
    (device rows cols : Nat) (before : World M) (d : Dense) (after : World M) : Prop where
  keep : Keep ops before after
  live : d.view.storage∈after.allocations
  fresh : ∀ old∈before.allocations,d.view.storage.separate old
  device : d.view.storage.owner=device
  aligned : d.view.storage.baseByte%16=0
  rows : d.rows=rows
  cols : d.cols=cols

/-- Ordinary context, Torch, allocator, and launch-setup contracts. Normalize
is the checked contiguous input view(-1,last): its storage keys alias the
original input, and its bookkeeping may register the same storage window. It says nothing about
L2 outputs, reductions, native arithmetic, or success of a kernel launch. -/
structure Library {M α β : Type} (ops : Ops M α β) (neg : Negation α) (rt : Runtime M) : Prop where
  enter : ∀ device s,Keep ops s (rt.enter device s).state
  leave : ∀ token s,Keep ops s (rt.leave token s).state
  normalize : ∀ input s d after,rt.normalize input s=.ok d after →
    input.backed → input.storage∈s.allocations → DirectInput input → Snapshot ops neg input s d after
  normalizeFailure : ∀ input s reason after,rt.normalize input s=.error reason after → Keep ops s after
  empty : ∀ device rows cols s d after,rt.empty device rows cols s=.ok d after → Allocated ops device rows cols s d after
  emptyFailure : ∀ device rows cols s reason after,rt.empty device rows cols s=.error reason after → Keep ops s after
  setup : ∀ p s,Keep ops s (rt.setup p s).state
  reshape : ∀ d shape out,rt.reshape d shape=some out →
    out.shape=shape ∧ out.storage=d.view.storage ∧ out.negative=Bool.false ∧
    out.numel=d.count ∧ ∀ i,i<d.count → out.key i=d.view.storage.key i

structure Prepared where
  snapshot : Dense
  output : Dense
  block : Nat

def Prepared.layout (q : Prepared) : Layout :=
  ⟨q.snapshot.view.storage,q.output.view.storage,q.snapshot.rows,q.snapshot.cols,q.block⟩

/-- Source order: checked direct input view, fresh output allocation and block
selection. The width condition is already guaranteed by the input guard. -/
def prepare {M α : Type} (rt : Runtime M) (call : Call α) (s : World M) : Result M Prepared :=
  (rt.normalize call.input s).bind fun x s1 =>
  (rt.empty x.view.storage.owner x.rows x.cols s1).bind fun y s2 =>
  if x.cols≤blockSize x.cols then .ok ⟨x,y,blockSize x.cols⟩ s2
  else .error "feature dimension exceeds 64KB" s2

/-- Computation is derived from complete source interpretation, through a
non-numerical observation adapter. Runtime setup selects only scheduling. -/
def launch {M α β γ : Type} [Inhabited α] [Inhabited γ] (ops : Ops M α β)
    (a : Source.Adapter α γ) (rt : Runtime M) (q : Prepared) (eps : γ) (s : World M) : Result M Unit :=
  let p := q.layout
  if p.rows=0 then .ok () s else
  (rt.setup p s).bind fun plan ready =>
  match plan with
  | .complete schedule _ => .ok () {ready with memory := Physical.run ops (a.compute p eps) p schedule ready.memory}
  | .interrupted completed storesDone _ _ reason =>
      .error reason {ready with memory := Physical.interrupted ops (a.compute p eps) p completed storesDone ready.memory}

/-- Restore the device context after either body outcome. A restoration error
is retained, and no rollback of completed private writes is asserted. -/
def context {M δ : Type} (rt : Runtime M) (device : Nat)
    (body : World M → Result M δ) (s : World M) : Result M δ :=
  (rt.enter device s).bind fun token entered =>
  let result := body entered
  match rt.leave token result.state with
  | .error reason after => .error reason after
  | .ok _ after => match result with
      | .ok value _ => .ok value after
      | .error reason _ => .error reason after

/-- The actual returned reshape uses the output allocated by this same call,
after leaving the CUDA device context, exactly as the Python wrapper does. -/
def execute {M α β γ : Type} [Inhabited α] [Inhabited γ] (ops : Ops M α β)
    (a : Source.Adapter α γ) (rt : Runtime M) (call : Call γ) (s : World M) : Result M View :=
  if call.selectedFP32 then
    (context rt call.device (fun entered =>
      (prepare rt call entered).bind fun q ready =>
      (launch ops a rt q call.eps ready).bind fun _ done => .ok q.output done) s).bind fun y final =>
    match rt.reshape y call.input.shape with
    | some out => .ok out final
    | none => .error "return reshape failed" final
  else .error "selected precision requires strided CUDA FP32" s

#axiomsClean block_covers
#axiomsClean block_power
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton1.Host
