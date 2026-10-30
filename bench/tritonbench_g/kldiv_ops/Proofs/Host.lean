import bench.tritonbench_g.kldiv_ops.Proofs.Storage

namespace VeriTile.Bench.Repairs.KldivOps.Host
open Storage VeriTile.Bench.Repairs.DiagSsmTriton.ByteMemory
noncomputable section

structure State where
  heap : Heap
  live : List Backing

inductive Result (A : Type) where
  | error (detail : String) (state : State)
  | ok (value : A) (state : State)

abbrev Action (A : Type) := State→Result A

def pure (value : A) : Action A := fun s => .ok value s

def bind (first : Action A) (next : A→Action B) : Action B := fun s =>
  match first s with
  | .error detail failed => .error detail failed
  | .ok value after => next value after

instance : Monad Action where
  pure := pure
  bind := bind

@[simp] theorem bind_error (e : String) (failed s : State) (next : A→Action B) :
    bind (fun _ => .error e failed) next s=.error e failed := rfl

def Ensures (post : A→State→Prop) : Result A→Prop
  | .error _ _ => True
  | .ok a s => post a s

theorem bind_ensures (first : Action A) (next : A→Action B) (s : State)
    (p : A→State→Prop) (q : B→State→Prop)
    (hfirst : Ensures p (first s))
    (hnext : ∀ a t,p a t → Ensures q (next a t)) :
    Ensures q (bind first next s) := by
  unfold bind
  cases h : first s with
  | error => trivial
  | ok a t => exact hnext a t (by simpa [h,Ensures] using hfirst)

inductive Reduction where
  | none | sum | mean | batchmean
  deriving DecidableEq,Repr

def Reduction.code : Reduction→Nat
  | .none => 0 | .sum => 1 | .mean => 2 | .batchmean => 3

def Reduction.outputShape (red : Reduction) (rows cols : Nat) :=
  if red=.none then [rows,cols] else [rows]

/-- Ordinary local library operations; each can fail with its actual state.
Kernel execution is a separate source-defined action, not a tensor service. -/
structure Services (Word : Type) where
  resolveConj : Tensor→Action Tensor
  resolveNeg : Tensor→Action Tensor
  contiguous : Tensor→Action Tensor
  clone : Tensor→Action Tensor
  zeros : List Nat→Nat→Action Tensor
  emptyContiguousLike : Tensor→Action Tensor
  copy : Tensor→Tensor→Action Tensor
  sum : Tensor→Bool→Action Tensor
  divNat : Tensor→Nat→Action Tensor
  scalarOne : Nat→Action Tensor
  equal : Tensor→Tensor→Action Bool
  multiply : Tensor→Tensor→Action Tensor
  configure : Nat→Nat→Action Unit

structure SourceActions (Word : Type) where
  forward : Nat→Nat→Tensor→Tensor→Tensor→Bool→Reduction→Word→Action Unit
  backward : Nat→Nat→Tensor→Tensor→Bool→Action Unit

def shape2 (t : Tensor) : Action (Nat×Nat) := fun s =>
  match t.shape with
  | [rows,cols] => .ok (rows,cols) s
  | _ => .error "two dimensions required" s

/-- Source-visible logical capacity validation, after the source's preparations.
No shape equality is imposed. A failed guard retains the actual prepared state. -/
def requireCapacity (t : Tensor) (count : Nat) (detail : String) : Action Unit := fun s =>
  if count ≤ t.numel then .ok () s else .error detail s

theorem capacity_success {a : Tensor} {count : Nat} {detail : String} {s t : State}
    (h : requireCapacity a count detail s=.ok () t) : count ≤ a.numel ∧ t=s := by
  unfold requireCapacity at h
  split at h
  next bound => cases h; exact ⟨bound,rfl⟩
  next => cases h

theorem capacity_failure (a : Tensor) (count : Nat) (detail : String) (s : State)
    (h : a.numel<count) : requireCapacity a count detail s=.error detail s := by
  simp [requireCapacity,show ¬count ≤ a.numel by omega]

def lookup (name : String) : Action Reduction := fun s =>
  if name="none" then .ok .none s
  else if name="sum" then .ok .sum s
  else if name="mean" then .ok .mean s
  else if name="batchmean" then .ok .batchmean s
  else .error "unknown reduction" s

def normalize (api : Services Word) (t : Tensor) : Action Tensor := do
  let c ← api.resolveConj t
  api.resolveNeg c

def prepare (api : Services Word) (t : Tensor) : Action Tensor := do
  let n ← normalize api t
  api.contiguous n

def snapshot (api : Services Word) (t : Tensor) : Action Tensor := do
  let n ← normalize api t
  api.clone n

/-- Exact source wrapper ordering. Setup/grid compilation lives in the
source action; configuration computes the original block/warp policy. -/
def forward (api : Services Word) (source : SourceActions Word)
    (prediction target : Tensor) (flag : Bool) (reduction : String) (eps : Word) : Action Tensor := do
  let (rows,cols) ← shape2 prediction
  let _ ← api.configure rows cols
  let red ← lookup reduction
  let out ← api.zeros (red.outputShape rows cols) prediction.device
  let py ← prepare api prediction
  let pt ← prepare api target
  let _ ← requireCapacity pt (rows*cols) "target has fewer elements than prediction"
  let _ ← source.forward rows cols py pt out flag red eps
  match red with
  | .none => pure out
  | .sum => api.sum out Bool.true
  | .mean =>
    let sum ← api.sum out Bool.false
    api.divNat sum (rows*cols)
  | .batchmean =>
    let sum ← api.sum out Bool.false
    api.divNat sum rows

def backwardFinish (api : Services Word) (grad destination work : Tensor) : Action Tensor := do
  let _ ← api.copy destination work
  let one ← api.scalarOne grad.device
  let isOne ← api.equal grad one
  if isOne then pure destination else api.multiply destination grad

def backward (api : Services Word) (source : SourceActions Word)
    (target grad destination : Tensor) (flag : Bool) : Action Tensor := do
  let (rows,cols) ← shape2 target
  let _ ← api.configure rows cols
  let target ← prepare api target
  let grad ← snapshot api grad
  let work ← api.emptyContiguousLike destination
  let _ ← requireCapacity work (rows*cols) "new_grads has fewer elements than target"
  let _ ← source.backward rows cols target work flag
  backwardFinish api grad destination work

/-- A failed post-write multiplication returns its failure state; writeback is
not rolled back. The entry cotangent has already been cloned by backward. -/
theorem multiply_failure_after_copy (api : Services Word) (grad dst work one : Tensor)
    (before copied compared failed : State) (e : String)
    (hc : api.copy dst work before=.ok dst copied)
    (ho : api.scalarOne grad.device copied=.ok one compared)
    (he : api.equal grad one compared=.ok Bool.false compared)
    (hm : api.multiply dst grad compared=.error e failed) :
    backwardFinish api grad dst work before=.error e failed := by
  simp [backwardFinish,Bind.bind,bind,hc,ho,he,hm]

theorem identity_after_copy (api : Services Word) (grad dst work one : Tensor)
    (before copied compared final : State)
    (hc : api.copy dst work before=.ok dst copied)
    (ho : api.scalarOne grad.device copied=.ok one compared)
    (he : api.equal grad one compared=.ok Bool.true final) :
    backwardFinish api grad dst work before=.ok dst final := by
  simp [backwardFinish,Bind.bind,bind,pure,hc,ho,he]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Host
