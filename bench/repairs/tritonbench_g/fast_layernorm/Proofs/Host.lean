import bench.repairs.tritonbench_g.fast_layernorm.Proofs.Storage
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.Compiler

/-! Ordinary local Torch contracts for contiguous, view, empty_like,
statistics allocation, saved context and device contexts. Historical generic
transfer/resolve/clone declarations remain available but are not called. Each
call has its actual failed state and a local arbitrary-Real shadow transport.
There is no kernel result or LayerNorm equation in the API. -/
namespace VeriTile.Bench.Repairs.FastLayernorm.Host
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
open scoped Classical

structure State (Store Metadata : Type) where
  memory : Storage.World Store
  metadata : Metadata
  owner : Nat
  streams : Nat → Nat
  live : Storage.Tensor → Prop
  ready : Backing → Nat → Prop

def world (api : Memory.API α Store Byte) (inputs : List Storage.Tensor) :
    VeriTile.Bench.Repairs.LayerNormFwd.Wrapper.World (State Store Metadata) Key Byte where
  observe s key := Storage.observe api s.memory key
  caller := Storage.caller inputs

structure Transition (api : Memory.API α Store Byte) (neg : α → α)
    (inputs : List Storage.Tensor) (before after : State Store Metadata) where
  shadow : Storage.World (Nat → ℝ) → Storage.World (Nat → ℝ)
  frame : ∀ key,Storage.caller inputs key → Storage.observe api after.memory key=Storage.observe api before.memory key
  realFrame : ∀ m key,Storage.caller inputs key → shadow m key.1 key.2=m key.1 key.2
  live : ∀ t,before.live t → after.live t
  values : ∀ t,before.live t → ∀ i,i<t.view.shape.prod →
    Storage.logical api neg after.memory t i=Storage.logical api neg before.memory t i
  realValues : ∀ m t,before.live t → ∀ i,i<t.view.shape.prod →
    Storage.realLogical (shadow m) t i=Storage.realLogical m t i
  streams : after.streams=before.streams
  ready : ∀ t stream,before.ready t stream → after.ready t stream

variable {api : Memory.API α Store Byte} {neg : α → α} {inputs : List Storage.Tensor}
  {before middle after : State Store Metadata}

def Transition.refl (s : State Store Metadata) : Transition api neg inputs s s where
  shadow := id
  frame _ _ := rfl
  realFrame _ _ _ := rfl
  live _ h := h
  values _ _ _ _ := rfl
  realValues _ _ _ _ _ := rfl
  streams := rfl
  ready _ _ h := h

def Transition.then (a : Transition api neg inputs before middle)
    (b : Transition api neg inputs middle after) : Transition api neg inputs before after where
  shadow := b.shadow ∘ a.shadow
  frame key hk := (b.frame key hk).trans (a.frame key hk)
  realFrame m key hk := (b.realFrame (a.shadow m) key hk).trans (a.realFrame m key hk)
  live t ht := b.live t (a.live t ht)
  values t ht i hi := (b.values t (a.live t ht) i hi).trans (a.values t ht i hi)
  realValues m t ht i hi := (b.realValues (a.shadow m) t (a.live t ht) i hi).trans (a.realValues m t ht i hi)
  streams := b.streams.trans a.streams
  ready t stream ht := b.ready t stream (a.ready t stream ht)

structure Effect (api : Memory.API α Store Byte) (neg : α → α)
    (inputs : List Storage.Tensor) (Metadata Value : Type) where
  call : Local (world (Metadata:=Metadata) api inputs) Value
  transition : ∀ s,Transition api neg inputs s (call.run s).1

/-- Logical element order and complete backing identity remain distinct. -/
structure CopyResult (api : Memory.API α Store Byte) (neg : α → α)
    (e : Effect api neg inputs Metadata Storage.Tensor) (t : Storage.Tensor)
    (s : State Store Metadata) (u : Storage.Tensor) : Prop where
  shape : u.view.shape=t.view.shape
  present : (e.call.run s).1.live u
  values : ∀ i,i<t.view.shape.prod →
    Storage.logical api neg (e.call.run s).1.memory u i=Storage.logical api neg s.memory t i
  realValues : ∀ m i,i<t.view.shape.prod →
    Storage.realLogical ((e.transition s).shadow m) u i=Storage.realLogical m t i
  ready : s.ready t.backing (s.streams t.view.device) →
    (e.call.run s).1.ready u.backing (s.streams u.view.device)

structure CloneResult (api : Memory.API α Store Byte) (neg : α → α)
    (e : Effect api neg inputs Metadata Dense) (t : Storage.Tensor)
    (s : State Store Metadata) (u : Dense) : Prop where
  shape : u.tensor.view.shape=t.view.shape
  owner : u.tensor.view.device=t.view.device
  present : (e.call.run s).1.live u.tensor
  fresh : ∀ old,s.live old → Storage.Disjoint u.tensor.backing old.backing
  callerFresh : ∀ old,old∈inputs → Storage.Disjoint u.tensor.backing old.backing
  values : ∀ i,i<t.view.shape.prod →
    Storage.logical api neg (e.call.run s).1.memory u.tensor i=Storage.logical api neg s.memory t i
  realValues : ∀ m i,i<t.view.shape.prod →
    Storage.realLogical ((e.transition s).shadow m) u.tensor i=Storage.realLogical m t i
  ready : s.ready t.backing (s.streams t.view.device) →
    (e.call.run s).1.ready u.tensor.backing (s.streams u.tensor.view.device)

/-- Ordinary checked contiguous() result. It need not allocate: an already
contiguous input retains its Tensor identity, backing and storage offset. -/
structure ContiguousResult (api : Memory.API α Store Byte) (neg : α → α)
    (e : Effect api neg inputs Metadata Dense) (t : Storage.Tensor)
    (s : State Store Metadata) (u : Dense) : Prop where
  shape : u.tensor.view.shape=t.view.shape
  owner : u.tensor.view.device=t.view.device
  present : (e.call.run s).1.live u.tensor
  values : ∀ i,i<t.view.shape.prod →
    Storage.logical api neg (e.call.run s).1.memory u.tensor i=Storage.logical api neg s.memory t i
  realValues : ∀ m i,i<t.view.shape.prod →
    Storage.realLogical ((e.transition s).shadow m) u.tensor i=Storage.realLogical m t i
  ready : s.ready t.backing (s.streams t.view.device) →
    (e.call.run s).1.ready u.tensor.backing (s.streams u.tensor.view.device)

structure ViewResult (api : Memory.API α Store Byte) (neg : α → α)
    (e : Effect api neg inputs Metadata Dense) (t : Dense) (shape : List Nat)
    (s : State Store Metadata) (u : Dense) : Prop where
  shape : u.tensor.view.shape=shape
  count : u.count=t.count
  backing : u.tensor.backing=t.tensor.backing
  present : (e.call.run s).1.live u.tensor
  values : ∀ i,i<t.count →
    Storage.logical api neg (e.call.run s).1.memory u.tensor i=Storage.logical api neg s.memory t.tensor i
  realValues : ∀ m i,i<t.count →
    Storage.realLogical ((e.transition s).shadow m) u.tensor i=Storage.realLogical m t.tensor i

structure AllocationResult (e : Effect api neg inputs Metadata Dense)
    (s : State Store Metadata) (u : Dense) : Prop where
  present : (e.call.run s).1.live u.tensor
  fresh : ∀ old,s.live old → Storage.Disjoint u.tensor.backing old.backing
  callerFresh : ∀ old,old∈inputs → Storage.Disjoint u.tensor.backing old.backing
  ready : (e.call.run s).1.ready u.tensor.backing (s.streams u.tensor.view.device)

structure SavedContext where
  identity : Nat
  (x w r mu : Dense)
  bias : Option Dense
  (inputShape weightShape : List Nat)
  biasShape : Option (List Nat)
  (weightDevice : Nat)
  biasDevice : Option Nat
  (rows columns block warps : Nat)
  epsilon : Compiler.Shared.HostScalar

structure APIHost (api : Memory.API α Store Byte) (neg : α → α)
    (inputs : List Storage.Tensor) (Metadata : Type) where
  enter : Nat → Effect api neg inputs Metadata Unit
  leave : Nat → Effect api neg inputs Metadata Unit
  enterOwner : ∀ owner s,((enter owner).call.run s).2=.ok () → ((enter owner).call.run s).1.owner=owner
  leaveOwner : ∀ owner s,((leave owner).call.run s).2=.ok () → ((leave owner).call.run s).1.owner=owner
  transfer : Nat → Storage.Tensor → Effect api neg inputs Metadata Storage.Tensor
  transferred : ∀ owner t s u,((transfer owner t).call.run s).2=.ok u →
    CopyResult api neg (transfer owner t) t s u ∧ u.view.device=owner
  resolve : Storage.Tensor → Effect api neg inputs Metadata Storage.Tensor
  resolved : ∀ t s u,((resolve t).call.run s).2=.ok u →
    CopyResult api neg (resolve t) t s u ∧ u.view.device=t.view.device ∧ u.view.negative=Bool.false
  clone : Storage.Tensor → Effect api neg inputs Metadata Dense
  cloned : ∀ t s u,((clone t).call.run s).2=.ok u → CloneResult api neg (clone t) t s u
  isContiguous : Storage.Tensor → Bool
  contiguous : Nat → Storage.Tensor → Effect api neg inputs Metadata Dense
  contiguousResult : ∀ device t s u,((contiguous device t).call.run s).2=.ok u →
    ContiguousResult api neg (contiguous device t) t s u ∧ u.tensor.view.device=device
  contiguousIdentity : ∀ device t s u,((contiguous device t).call.run s).2=.ok u →
    isContiguous t=Bool.true → u.tensor=t
  contiguousOwner : ∀ device t s,((contiguous device t).call.run s).1.owner=s.owner
  view : Dense → List Nat → Effect api neg inputs Metadata Dense
  viewed : ∀ t shape s u,((view t shape).call.run s).2=.ok u → ViewResult api neg (view t shape) t shape s u
  emptyLike : Dense → Effect api neg inputs Metadata Dense
  emptyLikeResult : ∀ t s u,((emptyLike t).call.run s).2=.ok u →
    AllocationResult (emptyLike t) s u ∧ u.tensor.view.shape=t.tensor.view.shape ∧ u.owner=t.owner
  emptyStats : Nat → Nat → Effect api neg inputs Metadata Dense
  emptyStatsResult : ∀ owner rows s u,((emptyStats owner rows).call.run s).2=.ok u →
    AllocationResult (emptyStats owner rows) s u ∧ u.tensor.view.shape=[rows] ∧ u.owner=owner
  context : Metadata → Nat → Option SavedContext
  save : SavedContext → Effect api neg inputs Metadata Unit
  saved : ∀ c s,((save c).call.run s).2=.ok () → context ((save c).call.run s).1.metadata c.identity=some c
  accessSaved : Nat → Effect api neg inputs Metadata SavedContext
  accessed : ∀ identity s c,((accessSaved identity).call.run s).2=.ok c → context s.metadata identity=some c
  /-- Ordinary autograd saved-handle/version contract, not a statistics law. -/
  savedIdentity : ∀ identity s c,((accessSaved identity).call.run s).2=.ok c → c.identity=identity
  transferOwner : ∀ owner t s,((transfer owner t).call.run s).1.owner=s.owner
  resolveOwner : ∀ t s,((resolve t).call.run s).1.owner=s.owner
  cloneOwner : ∀ t s,((clone t).call.run s).1.owner=s.owner
  viewOwner : ∀ t shape s,((view t shape).call.run s).1.owner=s.owner
  emptyLikeOwner : ∀ t s,((emptyLike t).call.run s).1.owner=s.owner
  emptyStatsOwner : ∀ owner rows s,((emptyStats owner rows).call.run s).1.owner=s.owner
  saveOwner : ∀ c s,((save c).call.run s).1.owner=s.owner
  accessOwner : ∀ identity s,((accessSaved identity).call.run s).1.owner=s.owner
  viewContext : ∀ t shape s,context ((view t shape).call.run s).1.metadata=context s.metadata
  leaveContext : ∀ owner s,context ((leave owner).call.run s).1.metadata=context s.metadata
  accessContext : ∀ identity s,context ((accessSaved identity).call.run s).1.metadata=context s.metadata
  /-- Successful access returns the actual live handles after normal autograd
  version checks. It does not identify their values with LayerNorm formulas. -/
  accessedPresent : ∀ identity s c,((accessSaved identity).call.run s).2=.ok c →
    ∀ t,t∈[c.x,c.w,c.r,c.mu]++c.bias.toList → ((accessSaved identity).call.run s).1.live t.tensor

/-- Admission checks and ordinary contiguous() on the same device. Forward
checks ensure an identity; only backward may require materializing dY. -/
def denseOn (lib : APIHost api neg inputs Metadata) (device : Nat) (t : Storage.Tensor) :
    Action (world (Metadata:=Metadata) api inputs) Dense :=
  call .contiguous (lib.contiguous device t).call

structure DenseRoute (lib : APIHost api neg inputs Metadata) (device : Nat)
    (original : Storage.Tensor) (initial : State Store Metadata) where
  output : Dense
  contiguousOK : ((lib.contiguous device original).call.run initial).2=.ok output

variable {lib : APIHost api neg inputs Metadata} {device : Nat} {original : Storage.Tensor}
  {initial : State Store Metadata}

def DenseRoute.final (_ : DenseRoute lib device original initial) :=
  ((lib.contiguous device original).call.run initial).1

def DenseRoute.transition (r : DenseRoute lib device original initial) : Transition api neg inputs initial r.final :=
  (lib.contiguous device original).transition initial

theorem call_ok (stage : Stage) (e : Effect api neg inputs Metadata Value)
    (s : State Store Metadata) (out : Value) :
    ((call stage e.call).run s).result=.ok out ↔ (e.call.run s).2=.ok out := by
  cases h : (e.call.run s).2 <;> simp [call,h,Except.mapError]

theorem dense_route (out : Dense)
    (ok : ((denseOn lib device original).run initial).result=.ok out) :
    ∃ r : DenseRoute lib device original initial,r.output=out ∧
      ((denseOn lib device original).run initial).state=r.final := by
  exact ⟨⟨out,(call_ok _ _ _ _).mp ok⟩,rfl,rfl⟩

theorem dense_metadata (r : DenseRoute lib device original initial) :
    r.output.tensor.view.shape=original.view.shape ∧ r.output.tensor.view.device=device ∧
    r.final.live r.output.tensor := by
  obtain ⟨info,owner⟩ := lib.contiguousResult device original initial r.output r.contiguousOK
  exact ⟨info.shape,owner,info.present⟩

theorem dense_values (r : DenseRoute lib device original initial) (i : Nat) (hi : i<original.view.shape.prod) :
    Storage.logical api neg r.final.memory r.output.tensor i=Storage.logical api neg initial.memory original i :=
  (lib.contiguousResult device original initial r.output r.contiguousOK).1.values i hi

theorem dense_real_values (r : DenseRoute lib device original initial)
    (m : Storage.World (Nat → ℝ)) (i : Nat) (hi : i<original.view.shape.prod) :
    Storage.realLogical (r.transition.shadow m) r.output.tensor i=Storage.realLogical m original i :=
  (lib.contiguousResult device original initial r.output r.contiguousOK).1.realValues m i hi

theorem dense_owner (r : DenseRoute lib device original initial) : r.final.owner=initial.owner :=
  lib.contiguousOwner device original initial

theorem dense_identity (r : DenseRoute lib device original initial)
    (contiguous : lib.isContiguous original=Bool.true) : r.output.tensor=original :=
  lib.contiguousIdentity device original initial r.output r.contiguousOK contiguous

#axiomsClean dense_owner
#axiomsClean dense_route
#axiomsClean dense_metadata
#axiomsClean dense_values
#axiomsClean dense_real_values
#axiomsClean dense_identity
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.Host
