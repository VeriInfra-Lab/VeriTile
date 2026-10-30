import bench.tritonbench_g.fused_recurrent_retention.Kernels
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
noncomputable section
def contiguous4 (d0 d1 d2 d3 : Nat) : Fin 4 → Nat :=
  ![max d1 1*max d2 1*max d3 1,max d2 1*max d3 1,max d3 1,1]

inductive ScalarType where | fp32 | other deriving DecidableEq

inductive DeviceKind where | cuda | other deriving DecidableEq

structure TensorObject where
  identity : Nat
  view : Tensor4
  dimensions : Fin 4 → Nat
  dtype : ScalarType
  deviceKind : DeviceKind
  conjugated : Bool
  negated : Bool
  requiresGrad : Bool

def TensorObject.RawFP32 (x : TensorObject) (current : Nat) : Prop :=
  x.dtype=.fp32 ∧ x.deviceKind=.cuda ∧ x.view.device=current ∧
    x.conjugated=Bool.false ∧ x.negated=Bool.false

def detach (x : TensorObject) (newIdentity : Nat) : TensorObject :=
  {x with identity := newIdentity,requiresGrad := Bool.false}

structure ForwardArguments where
  q : TensorObject
  k : TensorObject
  v : TensorObject
  seed : Option TensorObject
  detachedIdentity : Nat
  outputFinal : Bool

structure SavedContext where
  q : TensorObject
  k : TensorObject
  v : TensorObject
  initial : Option TensorObject
  outputFinal : Bool

def ForwardArguments.saved (x : ForwardArguments) : SavedContext :=
  ⟨x.q,x.k,x.v,x.seed.map (fun tensor => detach tensor x.detachedIdentity),x.outputFinal⟩

def SavedContext.buffers (c : SavedContext) (do_ : Tensor4) (final : Option Tensor4) : InputBuffers where
  q := c.q.view
  k := c.k.view
  v := c.v.view
  do_ := do_
  seed := (c.initial.getD c.q).view
  finalGrad := final.getD c.q.view

def ForwardArguments.buffers (x : ForwardArguments) : InputBuffers :=
  x.saved.buffers x.q.view none

structure ResultTensor where
  view : Tensor4
  dimensions : Fin 4 → Nat
  dtype : ScalarType

def resultTensor (storage device d0 d1 d2 d3 : Nat) : ResultTensor where
  view := ⟨storage,0,d0*d1*d2*d3,device,contiguous4 d0 d1 d2 d3⟩
  dimensions := ![d0,d1,d2,d3]
  dtype := .fp32

structure ForwardReturn (α : Type) where
  memory : Memory α
  output : ResultTensor
  finalState : Option ResultTensor
  context : SavedContext

structure BackwardReturn (α : Type) where
  memory : Memory α
  dq : ResultTensor
  dk : ResultTensor
  dv : ResultTensor
  initialGradient : Option Tensor4
  outputFlagGradient : Option Tensor4

def SavedContext.savedStorage (ctx : SavedContext) : Finset Nat :=
  {ctx.q.view.storage,ctx.k.view.storage,ctx.v.view.storage,(ctx.initial.getD ctx.q).view.storage}

def SavedContext.Unchanged (ctx : SavedContext) (forwardSnapshot backwardSnapshot : Memory α) : Prop :=
  ∀ p,p.1∈ctx.savedStorage → backwardSnapshot p=forwardSnapshot p
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention
