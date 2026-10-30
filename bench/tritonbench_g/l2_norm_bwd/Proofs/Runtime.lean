import bench.tritonbench_g.l2_norm_bwd.Proofs.Integer
import Mathlib.Data.Nat.Log

namespace VeriTile.Bench.Repairs.L2NormBwd.Host
open VeriTile.Triton FP.Structural Physical
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
open Classical
set_option maxHeartbeats 8000000

structure World (M : Type) where
  memory : M
  device : Nat
  streams : Nat → Nat
  live : Allocation → Prop

structure Context where
  object : Nat
  previous : Nat

structure Queue where
  owner : Nat
  stream : Nat

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

structure Frame {M α β : Type} (ops : Ops M α β) (kept : List Allocation) (a b : World M) : Prop where
  live : ∀ old,a.live old → b.live old
  reads : ∀ old∈kept,∀ key,old.contains key → ops.read b.memory key=ops.read a.memory key
  bytes : ∀ old∈kept,∀ owner address,old.covers owner address → ops.observe b.memory owner address=ops.observe a.memory owner address
  streams : b.streams=a.streams

theorem Frame.refl {M α β : Type} (ops : Ops M α β) (kept : List Allocation) (s : World M) : Frame ops kept s s :=
  ⟨fun _ h => h,fun _ _ _ _ => rfl,fun _ _ _ _ _ => rfl,rfl⟩

theorem Frame.trans {M α β : Type} {ops : Ops M α β} {kept : List Allocation} {a b c : World M}
    (ab : Frame ops kept a b) (bc : Frame ops kept b c) : Frame ops kept a c :=
  ⟨fun old h => bc.live old (ab.live old h),fun old ho key hk => (bc.reads old ho key hk).trans (ab.reads old ho key hk),
   fun old ho owner address hc => (bc.bytes old ho owner address hc).trans (ab.bytes old ho owner address hc),bc.streams.trans ab.streams⟩

theorem Frame.logical {M α β : Type} {ops : Ops M α β} {kept : List Allocation} {a b : World M}
    (h : Frame ops kept a b) (neg : Negation α) (v : View) (member : v.storage∈kept) (backed : v.backed)
    (i : Nat) (active : i<v.numel) : logical ops neg b.memory v i=logical ops neg a.memory v i := by
  unfold VeriTile.Bench.Repairs.SwigluTriton.Memory.logical
  rw [h.reads v.storage member _ (backed i active)]

/-- Every external operation is one ordinary Torch/runtime operation. Kernel
execution is deliberately absent; its memory effect is defined internally. -/
inductive Op : Type → Type
  | flatten (v : View) (cols : Nat) : Op Matrix
  | emptyLike (v : Matrix) : Op Matrix
  | resolve (v : Matrix) : Op Matrix
  | enter (owner : Nat) : Op Context
  | exit (token : Context) : Op Unit
  | setup (layout : Layout) (queue : Queue) : Op Unit
  | returnShape (v : Matrix) (shape : List Nat) : Op View

structure Runtime (M : Type) where
  perform : {A : Type} → Op A → World M → Result M A
  schedule : Layout → World M → List Nat

/-- Historical generic composition relation. The current view-only Library
below derives it from exact keys; no operation assumes its values field. -/
structure Copy {M α β : Type} (ops : Ops M α β) (neg : Negation α)
    (input output : View) (before after : World M) : Prop where
  size : output.numel=input.numel
  owner : output.storage.owner=input.storage.owner
  backed : output.backed
  live : after.live output.storage
  values : ∀ i,i < input.numel → logical ops neg after.memory output i=logical ops neg before.memory input i
  storage : output.storage=input.storage ∨ ∀ old,before.live old → output.storage.separate old

/-- A view operation preserves physical keys, negation and backing identity.
Its value relation is derived below, never supplied as a copied-input oracle. -/
structure ViewAlias {M : Type} (input output : View) (before after : World M) : Prop where
  size : output.numel=input.numel
  storage : output.storage=input.storage
  negative : output.negative=input.negative
  keys : ∀ i,i< input.numel → output.key i=input.key i
  backed : output.backed
  live : after.live output.storage
  state : after=before

theorem ViewAlias.copy {M α β : Type} (ops : Ops M α β) (neg : Negation α)
    {input output : View} {before after : World M} (h : ViewAlias input output before after) :
    Copy ops neg input output before after := by
  refine ⟨h.size,congrArg Allocation.owner h.storage,h.backed,h.live,?_,Or.inl h.storage⟩
  intro i hi
  simp only [VeriTile.Bench.Repairs.SwigluTriton.Memory.logical]
  rw [h.negative,h.keys i hi,h.state]

structure Library {M α β : Type} (ops : Ops M α β) (neg : Negation α) (rt : Runtime M) : Prop where
  frame : ∀ {A} (op : Op A) before kept,(∀ old∈kept,before.live old) →
    Frame ops kept before (rt.perform op before).state
  device : ∀ {A} (op : Op A) before,
    match op with
    | .enter _ | .exit _ => True
    | _ => (rt.perform op before).state.device=before.device
  flattenAlias : ∀ v cols before out after,rt.perform (.flatten v cols) before=.ok out after →
    0<cols ∧ out.cols=cols ∧ ViewAlias v out.view before after ∧ Integer.Metadata out
  emptyLike : ∀ v before out after,rt.perform (.emptyLike v) before=.ok out after →
    out.rows=v.rows ∧ out.cols=v.cols ∧ out.view.negative=Bool.false ∧ out.view.backed ∧ out.injective ∧
    after.live out.view.storage ∧ (∀ old,before.live old → out.view.storage.separate old) ∧ Integer.Metadata out ∧
    out.view.storage.owner=v.view.storage.owner
  resolveAlias : ∀ v before out after,rt.perform (.resolve v) before=.ok out after →
    out.rows=v.rows ∧ out.cols=v.cols ∧ out.view.negative=Bool.false ∧ ViewAlias v.view out.view before after ∧ Integer.Metadata out
  resolvePure : ∀ v s,v.view.backed → s.live v.view.storage → v.view.negative=Bool.false → Integer.Metadata v →
    rt.perform (.resolve v) s=.ok v s
  resolveIdentity : ∀ v before out after,rt.perform (.resolve v) before=.ok out after →
    v.view.negative=Bool.false → out=v ∧ after=before
  enter : ∀ owner before token after,rt.perform (.enter owner) before=.ok token after →
    token.previous=before.device ∧ after.memory=before.memory ∧ after.device=owner
  exit : ∀ token before after,rt.perform (.exit token) before=.ok () after →
    after.memory=before.memory ∧ after.device=token.previous
  setup : ∀ p queue before after,rt.perform (.setup p queue) before=.ok () after → after=before
  returnAlias : ∀ v shape before out after,rt.perform (.returnShape v shape) before=.ok out after →
    out.shape=shape ∧ ViewAlias v.view out before after
  schedule : ∀ p s,(rt.schedule p s).Perm (List.range p.x.rows)

/-- Compatibility facts for the accepted host composition; values follow from
alias keys and unchanged memory rather than normalization or a fresh copy. -/
theorem Library.flatten {M α β : Type} {ops : Ops M α β} {neg : Negation α} {rt : Runtime M}
    (lib : Library ops neg rt) (v : View) (cols : Nat) (before : World M) (out : Matrix) (after : World M)
    (h : rt.perform (.flatten v cols) before=.ok out after) :
    0<cols ∧ out.cols=cols ∧ Copy ops neg v out.view before after ∧ Integer.Metadata out := by
  obtain ⟨hp,hc,viewFacts,hm⟩:=lib.flattenAlias v cols before out after h
  exact ⟨hp,hc,viewFacts.copy ops neg,hm⟩

theorem Library.resolve {M α β : Type} {ops : Ops M α β} {neg : Negation α} {rt : Runtime M}
    (lib : Library ops neg rt) (v : Matrix) (before : World M) (out : Matrix) (after : World M)
    (h : rt.perform (.resolve v) before=.ok out after) :
    out.rows=v.rows ∧ out.cols=v.cols ∧ out.view.negative=Bool.false ∧ Copy ops neg v.view out.view before after ∧ Integer.Metadata out := by
  obtain ⟨hr,hc,hn,viewFacts,hm⟩:=lib.resolveAlias v before out after h
  exact ⟨hr,hc,hn,viewFacts.copy ops neg,hm⟩

theorem Library.returnShape {M α β : Type} {ops : Ops M α β} {neg : Negation α} {rt : Runtime M}
    (lib : Library ops neg rt) (v : Matrix) (shape : List Nat) (before : World M) (out : View) (after : World M)
    (h : rt.perform (.returnShape v shape) before=.ok out after) :
    out.shape=shape ∧ Copy ops neg v.view out before after := by
  obtain ⟨hs,viewFacts⟩:=lib.returnAlias v shape before out after h
  exact ⟨hs,viewFacts.copy ops neg⟩

structure Call where
  x : View
  dy : View

def Call.cols (call : Call) := call.dy.shape.getLastD 0
def Call.caller (call : Call) := [call.x.storage,call.dy.storage]

structure InputDomain {M : Type} (call : Call) (s : World M) : Prop where
  xBacked : call.x.backed
  dyBacked : call.dy.backed
  live : ∀ old∈call.caller,s.live old

def blockSize (cols : Nat) := min 16384 (2^Nat.clog 2 cols)

structure Request where
  x : Matrix
  dy : Matrix
  dx : Matrix
  block : Nat
  originalShape : List Nat

def prepareDY {M : Type} (rt : Runtime M) (dy : Matrix) (s : World M) : Result M Matrix :=
  .ok dy s

def normalize {M : Type} (rt : Runtime M) (call : Call) (s : World M) : Result M Request :=
  if call.dy.shape=[] then .error "DY last-dimension lookup failed" s else
  (rt.perform (.flatten call.x call.cols) s).bind fun x s1 =>
  (rt.perform (.flatten call.dy call.cols) s1).bind fun dy0 s2 =>
  (prepareDY rt dy0 s2).bind fun dy s3 =>
  (rt.perform (.emptyLike x) s3).bind fun dx s4 =>
  let block := blockSize x.cols
  if x.cols>block then .error "feature dimension exceeds 64KB" s4
  else if dy.view.numel<x.view.numel then .error "insufficient DY logical prefix" s4
  else if x.view.storage.owner≠dy.view.storage.owner then .error "different input devices" s4
  else .ok ⟨x,dy,dx,block,call.x.shape⟩ s4

/-- The actual final context exit runs after both body success and body error.
An exit failure carries that operation's actual state and replaces the error. -/
def finallyExit {M : Type} (rt : Runtime M) (token : Context) : Result M Unit → Result M Unit
  | .error reason s => (rt.perform (.exit token) s).bind fun _ after => .error reason after
  | .ok _ s => rt.perform (.exit token) s

def body {M α β : Type} (ops : Ops M α β) (alg : Reduction α) (rt : Runtime M)
    (q : Request) (eps : α) (s : World M) : Result M Unit :=
  (rt.perform (.resolve q.x) s).bind fun x s1 =>
  (rt.perform (.resolve q.dy) s1).bind fun dy s2 =>
  let p : Layout := ⟨x,dy,q.dx,q.block⟩
  (rt.perform (.setup p s2.queue) s2).bind fun _ s3 =>
  .ok () {s3 with memory:=Physical.run ops alg p eps (rt.schedule p s3) s3.memory}

def contextual {M α β : Type} (ops : Ops M α β) (alg : Reduction α) (rt : Runtime M)
    (q : Request) (eps : α) (s : World M) : Result M Unit :=
  (rt.perform (.enter q.x.view.storage.owner) s).bind fun token inside =>
    finallyExit rt token (body ops alg rt q eps inside)

def run {M α β : Type} (ops : Ops M α β) (alg : Reduction α) (rt : Runtime M)
    (call : Call) (eps : α) (s : World M) : Result M View :=
  (normalize rt call s).bind fun q prepared =>
  (contextual ops alg rt q eps prepared).bind fun _ computed =>
  rt.perform (.returnShape q.dx q.originalShape) computed

#specNonCircular Library avoiding [Physical.value,Physical.run,Tile.scalar,source]
#axiomsClean run
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormBwd.Host
