import bench.repairs.tritonbench_g.chunk_retention.Whole001.Proofs.SeparatePhysical
namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001
noncomputable section
open HostABI
namespace PhaseAPI

/-- Forward allocates only history, output and optional final-state storage.
Its caller set describes storage present at the forward call. -/
structure ForwardBuffers (caller : List Nat) where
  storage : Fin 3 → Nat
  fresh : ∀ i,storage i∉caller
  unique : Function.Injective storage
  strides : Fin 3 → List Nat

/-- Backward allocates its own dhistory/dQ/dK/dV-partials/dV storage at the
later call. Its caller set includes the cotangents that exist at that call. -/
structure BackwardBuffers (caller : List Nat) where
  storage : Fin 5 → Nat
  fresh : ∀ i,storage i∉caller
  unique : Function.Injective storage
  strides : Fin 5 → List Nat

def forwardReturn (args : CallMeta c) (a : ForwardBuffers caller) (memory : Memory α) : ForwardReturn α where
  memory := memory
  output := resultMeta (shapeValue c) args.currentDevice (a.storage 1) (a.strides 1)
  finalState := if args.finalRequested then some
    (resultMeta (shapeState c) args.currentDevice (a.storage 2) (a.strides 2)) else none
  savedQ := args.q
  savedK := args.k
  savedV := args.v
  savedHistory := resultMeta (shapeHistory c) args.currentDevice (a.storage 0) (a.strides 0)

def backwardReturn (args : CallMeta c) (a : BackwardBuffers caller) (memory : Memory α) : BackwardReturn α :=
  detachedBackward memory
    (resultMeta (shapeKey c) args.currentDevice (a.storage 1) (a.strides 1))
    (resultMeta (shapeKey c) args.currentDevice (a.storage 2) (a.strides 2))
    (resultMeta (shapeValue c) args.currentDevice (a.storage 4) (a.strides 4))

/-- Only optional scalar input regions may be virtually zero. -/
def OptionalBinding (b : Binding c) (r : Region) (m : Option TensorMeta) : Prop :=
  b.zeroInput r=m.isNone ∧ ∀ t,m=some t → b.base r=t.storage+t.offset

structure ForwardBinding (args : CallMeta c) (a : ForwardBuffers caller) (b : Binding c) : Prop where
  q : b.base .q=args.q.storage+args.q.offset
  k : b.base .k=args.k.storage+args.k.offset
  v : b.base .v=args.v.storage+args.v.offset
  qLive : b.zeroInput .q=Bool.false
  kLive : b.zeroInput .k=Bool.false
  vLive : b.zeroInput .v=Bool.false
  seed : OptionalBinding b .seed args.seed
  noIncoming : b.zeroInput .incoming=Bool.true
  noTerminal : b.zeroInput .terminal=Bool.true
  history : b.base .history=a.storage 0
  output : b.base .output=a.storage 1
  finalState : b.base .finalState=a.storage 2

structure Cotangents (args : CallMeta c) where
  incoming : Option TensorMeta
  terminal : Option TensorMeta
  incomingMeta : ∀ t,incoming=some t → t.dimensions=shapeValue c ∧ t.fp32=Bool.true ∧
    t.cuda=Bool.true ∧ t.resolved=Bool.true ∧ t.contiguous=Bool.true ∧ t.device=args.currentDevice
  terminalMeta : ∀ t,terminal=some t → t.dimensions=shapeState c ∧ t.fp32=Bool.true ∧
    t.cuda=Bool.true ∧ t.resolved=Bool.true ∧ t.contiguous=Bool.true ∧ t.device=args.currentDevice
  terminalAllowed : terminal.isSome=Bool.true → args.finalRequested=Bool.true

structure BackwardBinding (args : CallMeta c) (f : ForwardBuffers forwardCaller)
    (a : BackwardBuffers laterCaller) (g : Cotangents args) (b : Binding c) : Prop where
  q : b.base .q=args.q.storage+args.q.offset
  k : b.base .k=args.k.storage+args.k.offset
  v : b.base .v=args.v.storage+args.v.offset
  qLive : b.zeroInput .q=Bool.false
  kLive : b.zeroInput .k=Bool.false
  vLive : b.zeroInput .v=Bool.false
  noSeed : b.zeroInput .seed=Bool.true
  incoming : OptionalBinding b .incoming g.incoming
  terminal : OptionalBinding b .terminal g.terminal
  history : b.base .history=f.storage 0
  dhistory : b.base .dhistory=a.storage 0
  query : b.base .dq=a.storage 1
  key : b.base .dk=a.storage 2
  partialValue : b.base .dvPart=a.storage 3
  value : b.base .dv=a.storage 4

/-- Values read from actual returned storage handles at canonical active coordinates. -/
def tensorRead (m : Memory α) (t : TensorMeta) (offset : Nat) : α := m (t.storage+t.offset+offset)

theorem view_read (b : Binding c) (zero : α) (m : Memory α) (a : Address)
    (valid : a.Valid c) (writeOnly : a.region∉inputRegions) : b.view zero m a=m (b.address a) := by
  have live : b.zeroInput a.region=Bool.false := by
    cases h : b.zeroInput a.region
    · rfl
    · exact False.elim (writeOnly (b.zeroOnlyInputs a.region h))
  simp [Binding.view,valid,live]

structure ForwardPost (ops : Ops α) (c : Shape) (xs : Inputs α) (scale : α)
    (flag : Bool) (returned : ForwardReturn α) : Prop where
  output : ∀ h<c.heads,∀ t<c.T,∀ j<c.V,
    tensorRead returned.memory returned.output (c.valueOffset h t j)=Reference.output ops c xs scale h t j
  finalState : ∀ f,returned.finalState=some f → ∀ h<c.heads,∀ e<c.K,∀ j<c.V,
    tensorRead returned.memory f (c.stateOffset h e j)=Reference.finalState ops c xs h e j

structure BackwardPost (ops : Ops α) (c : Shape) (xs : Inputs α) (scale : α)
    (tree : SumTree) (returned : BackwardReturn α) : Prop where
  query : ∀ h<c.heads,∀ t<c.T,∀ e<c.K,
    tensorRead returned.memory returned.dq (c.keyOffset h t e)=Reference.dq ops c xs scale h t e
  key : ∀ h<c.heads,∀ t<c.T,∀ e<c.K,
    tensorRead returned.memory returned.dk (c.keyOffset h t e)=Reference.dk ops c xs scale h t e
  value : ∀ h<c.heads,∀ t<c.T,∀ j<c.V,
    tensorRead returned.memory returned.dv (c.valueOffset h t j)=Reference.dv ops c xs scale tree h t j
  detached : returned.initialGradient=none ∧ returned.flagGradient=none

theorem forward_handles (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree)
    (args : CallMeta c) (alloc : ForwardBuffers caller) (b : Binding c)
    (bound : ForwardBinding args alloc b) (before : Memory α) :
    ForwardPost ops c (inputsOfHeap (b.view ops.zero before)) scale args.finalRequested
      (forwardReturn args alloc (Separate.forwardPhysical ops c b scale tree args.finalRequested before)) := by
  have post:=Separate.forward_post ops c scale tree args.finalRequested (b.view ops.zero before)
  rw [← Separate.forward_physical] at post
  refine ⟨?_,?_⟩
  · intro h hh t ht j hj
    have v:=post.output h hh t ht j hj
    rw [view_read b ops.zero _ ⟨.output,h,0,t,j⟩ ⟨hh,rfl,ht,hj⟩ (by simp [inputRegions])] at v
    simpa only [tensorRead,forwardReturn,resultMeta,Binding.address,Address.offset,bound.output,Nat.add_zero] using v
  · intro f hf h hh e he j hj
    have flag : args.finalRequested=Bool.true := by
      cases hb:args.finalRequested <;> simp [forwardReturn,hb] at hf ⊢
    simp only [forwardReturn,flag,↓reduceIte,Option.some.injEq] at hf
    subst f
    have v:=post.finalState flag h hh e he j hj
    rw [view_read b ops.zero _ ⟨.finalState,h,0,e,j⟩ ⟨hh,rfl,he,hj⟩ (by simp [inputRegions])] at v
    simpa only [tensorRead,resultMeta,Binding.address,Address.offset,bound.finalState,Nat.add_zero] using v

theorem backward_handles (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree)
    (complete : tree.Complete c.keyBlocks) (args : CallMeta c)
    (falloc : ForwardBuffers forwardCaller) (alloc : BackwardBuffers laterCaller) (g : Cotangents args)
    (fb b : Binding c) (bound : BackwardBinding args falloc alloc g b) (before later : Memory α)
    (saved : Separate.Saved ops c scale tree args.finalRequested (fb.view ops.zero before) (b.view ops.zero later)) :
    BackwardPost ops c (Separate.joined (fb.view ops.zero before) (b.view ops.zero later)) scale tree
      (backwardReturn args alloc (Separate.backwardPhysical ops c b scale tree later)) := by
  have post:=Separate.backward_post ops c scale tree complete args.finalRequested _ _ saved
  rw [← Separate.backward_physical] at post
  refine ⟨?_,?_,?_,⟨rfl,rfl⟩⟩
  · intro h hh t ht e he
    have v:=post.query h hh t ht e he
    rw [view_read b ops.zero _ ⟨.dq,h,0,t,e⟩ ⟨hh,rfl,ht,he⟩ (by simp [inputRegions])] at v
    simpa only [tensorRead,backwardReturn,detachedBackward,resultMeta,Binding.address,Address.offset,bound.query,Nat.add_zero] using v
  · intro h hh t ht e he
    have v:=post.key h hh t ht e he
    rw [view_read b ops.zero _ ⟨.dk,h,0,t,e⟩ ⟨hh,rfl,ht,he⟩ (by simp [inputRegions])] at v
    simpa only [tensorRead,backwardReturn,detachedBackward,resultMeta,Binding.address,Address.offset,bound.key,Nat.add_zero] using v
  · intro h hh t ht j hj
    have v:=post.value h hh t ht j hj
    rw [view_read b ops.zero _ ⟨.dv,h,0,t,j⟩ ⟨hh,rfl,ht,hj⟩ (by simp [inputRegions])] at v
    simpa only [tensorRead,backwardReturn,detachedBackward,resultMeta,Binding.address,Address.offset,bound.value,Nat.add_zero] using v

#axiomsClean forward_handles
#axiomsClean backward_handles
end PhaseAPI
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001
