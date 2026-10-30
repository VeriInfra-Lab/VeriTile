import bench.repairs.tritonbench_g.kldiv_triton.Proofs.Storage
import bench.repairs.tritonbench_g.kldiv_triton.Proofs.Reduction
import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.Runtime

namespace VeriTile.Bench.Repairs.KldivTriton.Host
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open L2NormBwd.Physical (Matrix)
open L2NormBwd.Host (Context Queue)
open Physical (ForwardLayout BackwardLayout)
open FP.Equational (ReductionPlan)
open VeriTile.Bench.TritonBenchG.KldivTriton.FullMath (rounds)
noncomputable section
open Classical
set_option autoImplicit false
set_option maxHeartbeats 4000000

structure World (M : Type) extends L2NormBwd.Host.World M where
  gradEnabled : Bool

def World.queue {M : Type} (s : World M) : Queue := ⟨s.device,s.streams s.device⟩

inductive Result (M A : Type) where
  | error (reason : String) (state : World M)
  | ok (value : A) (state : World M)

def Result.state {M A : Type} : Result M A → World M
  | .error _ s => s | .ok _ s => s

def Result.bind {M A B : Type} (first : Result M A) (next : A → World M → Result M B) : Result M B :=
  match first with | .error reason s => .error reason s | .ok a s => next a s

def Ensures {M A : Type} (ok : A → World M → Prop) (error : String → World M → Prop) : Result M A → Prop
  | .error why s => error why s | .ok value s => ok value s

theorem bind_ok {M A B : Type} (first : Result M A) (next : A → World M → Result M B)
    (value : B) (after : World M) :
    first.bind next=.ok value after ↔ ∃ a before,first=.ok a before ∧ next a before=.ok value after := by
  cases first <;> simp [Result.bind]

/-- Local Torch scalar ports. dividePythonNat includes this exact wrapped
Python-integer conversion/dispatch and FP32 division, including zero and large
denominators. It is not replaced by Triton's division or Real division. -/
structure Arithmetic (α : Type) where
  zero : α
  one : α
  add : α → α → α
  mul : α → α → α
  equal : α → α → Bool
  dividePythonNat : α → Nat → α

def broadcastDim (a b : Nat) : Option Nat :=
  if a=b then some a else if a=1 then some b else if b=1 then some a else none

def broadcastRev : List Nat → List Nat → Option (List Nat)
  | [],ys => some ys
  | xs,[] => some xs
  | x::xs,y::ys => do
    let dim ← broadcastDim x y
    let rest ← broadcastRev xs ys
    pure (dim::rest)

def broadcastShape (a b : List Nat) : Option (List Nat) :=
  (broadcastRev a.reverse b.reverse).map List.reverse

def flatBroadcast : List Nat → List Nat → Nat → Nat
  | [],_,_ => 0
  | _,[],_ => 0
  | d::ds,_::os,i => (if d=1 then 0 else i/os.prod)*ds.prod + flatBroadcast ds os (i%os.prod)

def broadcastIndex (input output : List Nat) (i : Nat) : Nat :=
  flatBroadcast (List.replicate (output.length-input.length) 1 ++ input) output i

structure Summed (n : Nat) where
  view : View
  plan : ReductionPlan n

inductive Op : Type → Type
  | zeros (rows cols owner : Nat) : Op Matrix
  | empty (rows cols owner : Nat) : Op Matrix
  | resolveMatrix (v : Matrix) : Op Matrix
  | contiguous (v : Matrix) : Op Matrix
  | cloneMatrix (v : Matrix) : Op Matrix
  | resolve (v : View) : Op View
  | clone (v : View) : Op View
  | enter (owner : Nat) : Op Context
  | exit (token : Context) : Op Unit
  | enterNoGrad : Op Bool
  | exitNoGrad (previous : Bool) : Op Unit
  | setupForward (p : ForwardLayout) (queue : Queue) : Op Unit
  | setupBackward (p : BackwardLayout) (queue : Queue) : Op Unit
  | copyInto (destination source : View) : Op Unit
  | sum (v : View) : Op (Summed v.numel)
  | divide (v : View) (denominator : Nat) : Op View
  | scalarOne (owner : Nat) : Op View
  | equal (a b : View) : Op Bool
  | product (a b : View) : Op View

structure Runtime (M : Type) where
  perform : {A : Type} → Op A → World M → Result M A

abbrev Frame {M α β : Type} (ops : Ops M α β) (kept : List Allocation) (a b : World M) :=
  L2NormBwd.Host.Frame ops kept a.toWorld b.toWorld

abbrev Copy {M α β : Type} (ops : Ops M α β) (neg : Negation α)
    (input output : View) (before after : World M) :=
  L2NormBwd.Host.Copy ops neg input output before.toWorld after.toWorld

structure Fresh {M : Type} (v : View) (owner : Nat) (before after : World M) : Prop where
  owner : v.storage.owner=owner
  positive : v.negative=Bool.false
  backed : v.backed
  live : after.live v.storage
  separate : ∀ old,before.live old → v.storage.separate old

/-- Ordinary operation contracts only. All kernel execution is internal to
runForward/runBackward. Sum retains the actual addition tree; copy and product
state per-coordinate effects on the actual returned/supplied descriptors. -/
structure Library {M α β : Type} (ops : Ops M α β) (neg : Negation α)
    (arith : Arithmetic α) (rt : Runtime M) : Prop where
  frame : ∀ {A} (op : Op A) before kept,(∀ old∈kept,before.live old) →
    (match op with | .copyInto _ _ => True | _ => Frame ops kept before (rt.perform op before).state)
  live : ∀ {A} (op : Op A) before old,before.live old → (rt.perform op before).state.live old
  streams : ∀ {A} (op : Op A) before,(rt.perform op before).state.streams=before.streams
  device : ∀ {A} (op : Op A) before,
    match op with | .enter _ | .exit _ => True | _ => (rt.perform op before).state.device=before.device
  grad : ∀ {A} (op : Op A) before,
    match op with | .enterNoGrad | .exitNoGrad _ => True | _ => (rt.perform op before).state.gradEnabled=before.gradEnabled
  zeros : ∀ rows cols owner before out after,rt.perform (.zeros rows cols owner) before=.ok out after →
    out.rows=rows ∧ out.cols=cols ∧ out.colStride=1 ∧ out.injective ∧ Fresh out.view owner before after ∧
    (∀ i,i<out.view.numel → logical ops neg after.memory out.view i=arith.zero)
  empty : ∀ rows cols owner before out after,rt.perform (.empty rows cols owner) before=.ok out after →
    out.rows=rows ∧ out.cols=cols ∧ out.colStride=1 ∧ out.injective ∧ Fresh out.view owner before after
  resolveMatrix : ∀ v before out after,rt.perform (.resolveMatrix v) before=.ok out after →
    out.rows=v.rows ∧ out.cols=v.cols ∧ out.view.negative=Bool.false ∧ Copy ops neg v.view out.view before after
  contiguous : ∀ v before out after,rt.perform (.contiguous v) before=.ok out after →
    out.rows=v.rows ∧ out.cols=v.cols ∧ Index.UnitColumns out ∧ out.view.negative=v.view.negative ∧ Copy ops neg v.view out.view before after
  cloneMatrix : ∀ v before out after,rt.perform (.cloneMatrix v) before=.ok out after →
    out.rows=v.rows ∧ out.cols=v.cols ∧ out.colStride=1 ∧ out.view.negative=Bool.false ∧
    Copy ops neg v.view out.view before after ∧ (∀ old,before.live old → out.view.storage.separate old)
  resolve : ∀ v before out after,rt.perform (.resolve v) before=.ok out after →
    out.shape=v.shape ∧ out.negative=Bool.false ∧ Copy ops neg v out before after
  clone : ∀ v before out after,rt.perform (.clone v) before=.ok out after →
    out.shape=v.shape ∧ out.negative=Bool.false ∧ Copy ops neg v out before after ∧
    (∀ old,before.live old → out.storage.separate old)
  enter : ∀ owner before token after,rt.perform (.enter owner) before=.ok token after →
    token.previous=before.device ∧ after.memory=before.memory ∧ after.device=owner
  exit : ∀ token before after,rt.perform (.exit token) before=.ok () after →
    after.memory=before.memory ∧ after.device=token.previous
  enterNoGrad : ∀ before token after,rt.perform .enterNoGrad before=.ok token after →
    token=before.gradEnabled ∧ after.memory=before.memory ∧ after.gradEnabled=Bool.false
  exitNoGrad : ∀ token before after,rt.perform (.exitNoGrad token) before=.ok () after →
    after.memory=before.memory ∧ after.gradEnabled=token
  setupForward : ∀ p queue before after,rt.perform (.setupForward p queue) before=.ok () after → after=before
  setupBackward : ∀ p queue before after,rt.perform (.setupBackward p queue) before=.ok () after → after=before
  copyInto : ∀ dst src before after,rt.perform (.copyInto dst src) before=.ok () after →
    src.numel=dst.numel ∧ (∀ i,i<dst.numel → logical ops neg after.memory dst i=logical ops neg before.memory src i)
  copyFrame : ∀ dst src before,
    (∀ owner address,(∀ i,i<dst.numel → ¬(dst.key i).covers owner address) →
      ops.observe (rt.perform (.copyInto dst src) before).state.memory owner address=ops.observe before.memory owner address) ∧
    (∀ key,(∀ i,i<dst.numel → key.disjoint (dst.key i)) →
      ops.read (rt.perform (.copyInto dst src) before).state.memory key=ops.read before.memory key)
  sum : ∀ v before out after,rt.perform (.sum v) before=.ok out after →
    out.view.shape=[] ∧ Fresh out.view v.storage.owner before after ∧
    logical ops neg after.memory out.view 0=
      HostReduction.eval arith.add arith.zero (fun i => logical ops neg before.memory v i.val) out.plan.tree
  divide : ∀ v denominator before out after,rt.perform (.divide v denominator) before=.ok out after →
    v.shape=[] ∧ out.shape=[] ∧ Fresh out v.storage.owner before after ∧
    logical ops neg after.memory out 0=arith.dividePythonNat (logical ops neg before.memory v 0) denominator
  scalarOne : ∀ owner before out after,rt.perform (.scalarOne owner) before=.ok out after →
    out.shape=[] ∧ Fresh out owner before after ∧ logical ops neg after.memory out 0=arith.one
  equal : ∀ a b before value after,rt.perform (.equal a b) before=.ok value after →
    value=decide (a.shape=b.shape ∧ ∀ i,i<a.numel →
      arith.equal (logical ops neg before.memory a i) (logical ops neg before.memory b i)=Bool.true)
  product : ∀ a b before out after,rt.perform (.product a b) before=.ok out after →
    broadcastShape a.shape b.shape=some out.shape ∧ Fresh out a.storage.owner before after ∧
    (∀ i,i<out.numel → broadcastIndex a.shape out.shape i<a.numel ∧ broadcastIndex b.shape out.shape i<b.numel ∧
      logical ops neg after.memory out i=arith.mul
        (logical ops neg before.memory a (broadcastIndex a.shape out.shape i))
        (logical ops neg before.memory b (broadcastIndex b.shape out.shape i)))

def blockSize (cols : Nat) : Nat := max 1 (min 16384 (2^Nat.clog 2 cols))
theorem block_positive (cols : Nat) : 0<blockSize cols := by unfold blockSize; omega

def modeCode : Math.Mode → Nat
  | .none => 0 | .sum => 1 | .mean => 2 | .batchmean => 3

structure ForwardCall where
  prediction : Matrix
  target : Matrix
  flag : Bool
  mode : Math.Mode

structure BackwardCall where
  input : Matrix
  target : Matrix
  gradient : View
  flag : Bool

/-- The actual finally operation executes after both success and error. Its
failure carries that operation's state, rather than rolling back a prefix. -/
def finallyDevice {M A : Type} (rt : Runtime M) (token : Context) : Result M A → Result M A
  | .error reason s => (rt.perform (.exit token) s).bind fun _ after => .error reason after
  | .ok value s => (rt.perform (.exit token) s).bind fun _ after => .ok value after

def finallyGrad {M A : Type} (rt : Runtime M) (token : Bool) : Result M A → Result M A
  | .error reason s => (rt.perform (.exitNoGrad token) s).bind fun _ after => .error reason after
  | .ok value s => (rt.perform (.exitNoGrad token) s).bind fun _ after => .ok value after

def finishForward {M : Type} (rt : Runtime M) (call : ForwardCall) (output : Matrix) (s : World M) : Result M View :=
  match call.mode with
  | .none => .ok output.view s
  | .sum => (rt.perform (.sum output.view) s).bind fun summed after => .ok summed.view after
  | .mean => (rt.perform (.sum output.view) s).bind fun summed after =>
      rt.perform (.divide summed.view (call.prediction.rows*call.prediction.cols)) after
  | .batchmean => (rt.perform (.sum output.view) s).bind fun summed after =>
      rt.perform (.divide summed.view call.prediction.rows) after

def prepareMatrix {M : Type} (rt : Runtime M) (input : Matrix) (snapshot : Bool) (s : World M) : Result M Matrix :=
  (rt.perform (.resolveMatrix input) s).bind fun resolved s1 =>
  rt.perform (if snapshot then .cloneMatrix resolved else .contiguous resolved) s1

def prepareGradient {M : Type} (rt : Runtime M) (input : View) (s : World M) : Result M View :=
  (rt.perform (.resolve input) s).bind fun resolved s1 => rt.perform (.clone resolved) s1

def forwardBody {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (adapter : Storage.Adapter α κ)
    (rt : Runtime M) (call : ForwardCall) (output : Matrix) (s : World M) : Result M View :=
  (prepareMatrix rt call.prediction Bool.false s).bind fun y s2 =>
  (prepareMatrix rt call.target Bool.false s2).bind fun t s4 =>
  let p : ForwardLayout := ⟨y,t,output,blockSize call.prediction.cols,call.flag,modeCode call.mode⟩
  (rt.perform (.setupForward p s4.queue) s4).bind fun _ s5 =>
  finishForward rt call output {s5 with memory:=Storage.forwardKernel ops adapter s5.memory p (block_positive _)}

def runForward {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (adapter : Storage.Adapter α κ)
    (rt : Runtime M) (call : ForwardCall) (s : World M) : Result M View :=
  if call.target.rows≠call.prediction.rows ∨ call.target.cols≠call.prediction.cols then .error "shape mismatch" s
  else if call.target.view.storage.owner≠call.prediction.view.storage.owner then .error "device mismatch" s
  else
    let cols := if call.mode=.none then call.prediction.cols else rounds call.prediction.cols (blockSize call.prediction.cols)
    (rt.perform (.zeros call.prediction.rows cols call.prediction.view.storage.owner) s).bind fun output s1 =>
    (rt.perform (.enter call.prediction.view.storage.owner) s1).bind fun token inside =>
    finallyDevice rt token (forwardBody ops adapter rt call output inside)

def copyNoGrad {M : Type} (rt : Runtime M) (destination source : View) (s : World M) : Result M Unit :=
  (rt.perform .enterNoGrad s).bind fun token inside =>
  finallyGrad rt token (rt.perform (.copyInto destination source) inside)

def finishBackward {M : Type} (rt : Runtime M) (call : BackwardCall) (gradient : View) (s : World M) : Result M View :=
  (rt.perform (.scalarOne gradient.storage.owner) s).bind fun one s1 =>
  (rt.perform (.equal gradient one) s1).bind fun equal s2 =>
  if equal then .ok call.input.view s2 else rt.perform (.product call.input.view gradient) s2

def backwardBody {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (adapter : Storage.Adapter α κ)
    (rt : Runtime M) (call : BackwardCall) (s : World M) : Result M View :=
  (prepareMatrix rt call.target Bool.true s).bind fun target s2 =>
  (prepareGradient rt call.gradient s2).bind fun gradient s4 =>
  (rt.perform (.empty call.input.rows call.input.cols call.input.view.storage.owner) s4).bind fun work s5 =>
  let p : BackwardLayout := ⟨target,work,blockSize call.input.cols,call.flag⟩
  (rt.perform (.setupBackward p s5.queue) s5).bind fun _ s6 =>
  let computed := {s6 with memory:=Storage.backwardKernel ops adapter s6.memory p (block_positive _)}
  (copyNoGrad rt call.input.view work.view computed).bind fun _ copied =>
  finishBackward rt call gradient copied

def runBackward {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (adapter : Storage.Adapter α κ)
    (rt : Runtime M) (call : BackwardCall) (s : World M) : Result M View :=
  if call.target.rows≠call.input.rows ∨ call.target.cols≠call.input.cols then .error "shape mismatch" s
  else if call.target.view.storage.owner≠call.input.view.storage.owner then .error "device mismatch" s
  else if ¬Layout.Writable call.input.rows call.input.cols call.input.rowStride call.input.colStride then .error "overlapping destination" s
  else (rt.perform (.enter call.input.view.storage.owner) s).bind fun token inside =>
    finallyDevice rt token (backwardBody ops adapter rt call inside)

#specNonCircular Library avoiding [Storage.forwardValue,Storage.backwardValue,Storage.forwardKernel,Storage.backwardKernel,Tile.sourceRun,Backward.sourceRun]
#axiomsClean runForward
#axiomsClean runBackward
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Host
