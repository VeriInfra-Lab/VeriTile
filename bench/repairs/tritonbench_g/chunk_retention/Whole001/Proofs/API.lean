import bench.repairs.tritonbench_g.chunk_retention.Whole001.Proofs.Post
namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001
noncomputable section
open HostABI

/-- Source-visible normalized handles. The original @contiguous wrapper is a
coordinate-preserving copy when needed, before these handles are validated. -/
structure CallMeta (c : Shape) where
  q : TensorMeta
  k : TensorMeta
  v : TensorMeta
  seed : Option TensorMeta
  currentDevice : Nat
  finalRequested : Bool
  normalizedQ : q.dimensions=shapeKey c ∧ q.fp32=Bool.true ∧ q.cuda=Bool.true ∧
    q.resolved=Bool.true ∧ q.contiguous=Bool.true ∧ q.device=currentDevice
  normalizedK : k.dimensions=shapeKey c ∧ k.fp32=Bool.true ∧ k.cuda=Bool.true ∧
    k.resolved=Bool.true ∧ k.contiguous=Bool.true ∧ k.device=currentDevice
  normalizedV : v.dimensions=shapeValue c ∧ v.fp32=Bool.true ∧ v.cuda=Bool.true ∧
    v.resolved=Bool.true ∧ v.contiguous=Bool.true ∧ v.device=currentDevice
  seedMeta : ∀ s,seed=some s → s.dimensions=shapeState c ∧ s.fp32=Bool.true ∧
    s.cuda=Bool.true ∧ s.resolved=Bool.true ∧ s.contiguous=Bool.true ∧ s.device=currentDevice

/-- Fresh allocations are ordinary torch new_empty/empty_like/sum contracts.
Actual singleton/empty strides are retained; the public values use canonical
logical coordinates. No input/input separation premise is required. -/
structure Allocations (c : Shape) (caller : List Nat) where
  storage : Fin 8 → Nat
  fresh : ∀ i,storage i∉caller
  unique : Function.Injective storage
  strides : Fin 8 → List Nat

def resultMeta (dims : List Nat) (device storage : Nat) (strides : List Nat) : TensorMeta :=
  ⟨dims,strides,storage,0,dims.prod,device,Bool.true,Bool.true,Bool.true,Bool.true⟩

def forwardReturn (args : CallMeta c) (alloc : Allocations c caller) (memory : Memory α) : ForwardReturn α where
  memory := memory
  output := resultMeta (shapeValue c) args.currentDevice (alloc.storage 1) (alloc.strides 1)
  finalState := if args.finalRequested then
    some (resultMeta (shapeState c) args.currentDevice (alloc.storage 2) (alloc.strides 2)) else none
  savedQ := args.q
  savedK := args.k
  savedV := args.v
  savedHistory := resultMeta (shapeHistory c) args.currentDevice (alloc.storage 0) (alloc.strides 0)

def backwardReturn (args : CallMeta c) (alloc : Allocations c caller) (memory : Memory α) : BackwardReturn α :=
  detachedBackward memory
    (resultMeta (shapeKey c) args.currentDevice (alloc.storage 4) (alloc.strides 4))
    (resultMeta (shapeKey c) args.currentDevice (alloc.storage 5) (alloc.strides 5))
    (resultMeta (shapeValue c) args.currentDevice (alloc.storage 7) (alloc.strides 7))

def APIPost (args : CallMeta c) (alloc : Allocations c caller) (forwardMemory backwardMemory : Memory α) : Prop :=
  let f:=forwardReturn args alloc forwardMemory
  let b:=backwardReturn args alloc backwardMemory
  f.savedQ=args.q ∧ f.savedK=args.k ∧ f.savedV=args.v ∧
  f.output.dimensions=shapeValue c ∧ f.output.device=args.currentDevice ∧ f.output.fp32=Bool.true ∧
  f.savedHistory.dimensions=shapeHistory c ∧ f.finalState.isSome=args.finalRequested ∧
  b.dq.dimensions=shapeKey c ∧ b.dk.dimensions=shapeKey c ∧ b.dv.dimensions=shapeValue c ∧
  b.initialGradient=none ∧ b.flagGradient=none ∧
  f.output.storage∉caller ∧ b.dq.storage∉caller ∧ b.dk.storage∉caller ∧ b.dv.storage∉caller

theorem api_post (args : CallMeta c) (alloc : Allocations c caller) (f b : Memory α) :
    APIPost args alloc f b := by
  unfold APIPost forwardReturn backwardReturn detachedBackward resultMeta
  refine ⟨rfl,rfl,rfl,rfl,rfl,rfl,rfl,?_,rfl,rfl,rfl,rfl,rfl,
    alloc.fresh 1,alloc.fresh 4,alloc.fresh 5,alloc.fresh 7⟩
  cases args.finalRequested <;> rfl

/-- Backward has its own cotangent snapshot. Autograd preserves the saved
normalized Q/K/V and history handles/versions; initial_state is detached and
is not saved by the backward context. Its later mutation is irrelevant. -/
structure BackwardSnapshot (α : Type) (args : CallMeta c) (saved : ForwardReturn α) where
  memory : Memory α
  incoming : Option TensorMeta
  terminal : Option TensorMeta
  savedIdentity : saved.savedQ=args.q ∧ saved.savedK=args.k ∧ saved.savedV=args.v
  terminalAllowed : terminal.isSome=Bool.true → args.finalRequested=Bool.true
  savedWord : ∀ p, p<saved.savedHistory.capacity →
    memory (saved.savedHistory.storage+p)=saved.memory (saved.savedHistory.storage+p)

#axiomsClean api_post
#axiomsClean HostABI.initial_detach
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001
