import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.Contract
import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.Calls
import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.CompleteFrame
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
noncomputable section
namespace Host
open Admission

/-- Allocator metadata and object identities are observations of actual returned
handles. Structural scopes are derived below, rather than supplied by callers. -/
structure Forward (α : Type) (s : Shape) (caller : Set Nat) where
  args : ForwardArguments
  current : Nat
  bases : Bases
  allocation : Allocation 3 caller
  identities : Fin 3 → Nat
  requiresGrad : Fin 3 → Bool
  inputs : ∀ id,id∈args.buffers.storageIds → id∈caller
  before : Memory α

structure Backward (α : Type) (s : Shape) (ctx : SavedContext) (snapshot : Memory α)
    (current : Nat) (caller : Set Nat) where
  do_ : TensorObject
  final : Option TensorObject
  gradEnabled : Bool
  bases : Bases
  allocation : Allocation 6 caller
  identities : Fin 6 → Nat
  requiresGrad : Fin 6 → Bool
  inputs : ∀ id,id∈(ctx.buffers do_.view (final.map TensorObject.view)).storageIds → id∈caller
  before : Memory α
  saved : ctx.Unchanged snapshot before

def Forward.allocations (x : Forward α s caller) : List (Nat × Nat) :=
  [(x.allocation.storage 0,s.NK*rowCount s s.DV)]++
    if x.args.outputFinal then [(x.allocation.storage 1,s.BH*s.DK*s.DV)] else []

def Backward.allocations (x : Backward α s ctx snapshot current caller) : List (Nat × Nat) :=
  [(x.allocation.storage 0,s.NV*rowCount s s.DK),
   (x.allocation.storage 1,s.NV*rowCount s s.DK),(x.allocation.storage 2,s.NK*rowCount s s.DV)]

def Forward.check (x : Forward α s caller) : Bool :=
  forwardCheck s x.args x.current x.bases && allocationCheck x.bases
    (pointerRadius .forward x.args.seed.isSome x.args.outputFinal s x.args.buffers) x.allocations

def Backward.check (x : Backward α s ctx snapshot current caller) : Bool :=
  backwardCheck s ctx x.do_ x.final current x.bases x.gradEnabled && allocationCheck x.bases
    (pointerRadius .backward ctx.initial.isSome x.final.isSome s (ctx.buffers x.do_.view (x.final.map TensorObject.view))) x.allocations

def Forward.call (x : Forward α s caller) (ok : x.check=Bool.true) : ForwardCall α s caller where
  args := x.args
  currentDevice := x.current
  scope := checked_forward s x.args x.current x.bases (by have h := ok; simp only [Forward.check,Bool.and_eq_true] at h; exact h.1)
  allocation := x.allocation
  inputs := x.inputs
  before := x.before

def Backward.call (x : Backward α s ctx snapshot current caller) (ok : x.check=Bool.true) :
    BackwardCall α s ctx snapshot current caller where
  do_ := x.do_
  final := x.final
  scope := checked_backward s ctx x.do_ x.final current x.bases x.gradEnabled (by have h := ok; simp only [Backward.check,Bool.and_eq_true] at h; exact h.1)
  allocation := x.allocation
  inputs := x.inputs
  before := x.before
  saved := x.saved

/-- Tensor-object identity belongs to the actual allocator result; it is not
identified with its backing storage number. -/
def object (id : Nat) (requiresGrad : Bool) (x : ResultTensor) : TensorObject :=
  ⟨id,x.view,x.dimensions,x.dtype,.cuda,Bool.false,Bool.false,requiresGrad⟩

structure ForwardResult (α : Type) where
  source : ForwardReturn α
  output : TensorObject
  final : Option TensorObject

structure BackwardResult (α : Type) where
  source : BackwardReturn α
  gradients : TensorObject × TensorObject × TensorObject × Option TensorObject × Option TensorObject

def forward (a : Arithmetic α) (p : PlanBinding s) (x : Forward α s caller) : Option (ForwardResult α) :=
  if ok : x.check=Bool.true then
    let ret := (x.call ok).sourceRun a p
    some ⟨ret,object (x.identities 2) (x.requiresGrad 2) ret.output,ret.finalState.map (object (x.identities 1) (x.requiresGrad 1))⟩
  else none

def backward (a : Arithmetic α) (p : PlanBinding s) (x : Backward α s ctx snapshot current caller) : Option (BackwardResult α) :=
  if ok : x.check=Bool.true then
    let ret := (x.call ok).sourceRun a p
    some ⟨ret,(object (x.identities 3) (x.requiresGrad 3) ret.dq,object (x.identities 4) (x.requiresGrad 4) ret.dk,object (x.identities 5) (x.requiresGrad 5) ret.dv,none,none)⟩
  else none

def forwardMemory (a : Arithmetic α) (p : PlanBinding s) (x : Forward α s caller) : Memory α :=
  ((forward a p x).map (fun r=>r.source.memory)).getD x.before

def backwardMemory (a : Arithmetic α) (p : PlanBinding s) (x : Backward α s ctx snapshot current caller) : Memory α :=
  ((backward a p x).map (fun r=>r.source.memory)).getD x.before

theorem forward_rejected (a : Arithmetic α) (p : PlanBinding s) (x : Forward α s caller)
    (reject : x.check=Bool.false) : forward a p x=none ∧ forwardMemory a p x=x.before := by
  simp [forward,forwardMemory,reject]

theorem backward_rejected (a : Arithmetic α) (p : PlanBinding s) (x : Backward α s ctx snapshot current caller)
    (reject : x.check=Bool.false) : backward a p x=none ∧ backwardMemory a p x=x.before := by
  simp [backward,backwardMemory,reject]

theorem forward_frame (a : Arithmetic α) (p : PlanBinding s) (x : Forward α s caller) (address : Address)
    (off : ¬CompleteFrame.forwardWrites s x.allocation x.args.outputFinal address) :
    forwardMemory a p x address=x.before address := by
  by_cases ok : x.check=Bool.true
  · simp only [forwardMemory,forward,dif_pos ok,Option.map_some,Option.getD_some,
      ForwardCall.sourceRun,Forward.call]
    exact CompleteFrame.forward _ _ _ _ _ _ _ _ _ off
  · simp [forwardMemory,forward,ok]

theorem backward_frame (a : Arithmetic α) (p : PlanBinding s) (x : Backward α s ctx snapshot current caller) (address : Address)
    (off : ¬CompleteFrame.backwardWrites s x.allocation address) :
    backwardMemory a p x address=x.before address := by
  by_cases ok : x.check=Bool.true
  · simp only [backwardMemory,backward,dif_pos ok,Option.map_some,Option.getD_some,
      BackwardCall.sourceRun,Backward.call]
    exact CompleteFrame.backward _ _ _ _ _ _ _ _ _ off
  · simp [backwardMemory,backward,ok]

theorem forward_bytes (a : Arithmetic Numeric.Word) (p : PlanBinding s)
    (x : Forward Numeric.Word s caller) (address : Nat × Int)
    (off : ¬CompleteFrame.forwardWrites s x.allocation x.args.outputFinal (address.1,address.2/4)) :
    CompleteFrame.byteView (forwardMemory a p x) address=CompleteFrame.byteView x.before address := by
  simp only [CompleteFrame.byteView,forward_frame a p x _ off]

theorem backward_bytes (a : Arithmetic Numeric.Word) (p : PlanBinding s)
    (x : Backward Numeric.Word s ctx snapshot current caller) (address : Nat × Int)
    (off : ¬CompleteFrame.backwardWrites s x.allocation (address.1,address.2/4)) :
    CompleteFrame.byteView (backwardMemory a p x) address=CompleteFrame.byteView x.before address := by
  simp only [CompleteFrame.byteView,backward_frame a p x _ off]

#axiomsClean forward_bytes
#axiomsClean backward_bytes
#axiomsClean Forward.call
#axiomsClean Backward.call
#axiomsClean forward_rejected
#axiomsClean backward_rejected
#axiomsClean forward_frame
#axiomsClean backward_frame
end Host
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention
