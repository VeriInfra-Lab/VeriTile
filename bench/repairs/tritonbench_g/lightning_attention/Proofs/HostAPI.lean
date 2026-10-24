import bench.repairs.tritonbench_g.lightning_attention.Proofs.HostPhysical
import bench.repairs.tritonbench_g.lightning_attention.Proofs.PhasePrograms
namespace VeriTile.Bench.Repairs.LightningAttention.Host
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
variable {α : Type} [Inhabited α]
/-- Selected CUDA launch profile. The Python wrapper uses the caller's current
CUDA device and current stream; it does not switch or restore devices. -/
structure LaunchEnvironment where
  cudaBackend : Bool
  currentDevice : Nat
  currentStream : Nat
  streamDevice : Nat
def LaunchContract (device : Nat) (e : LaunchEnvironment) : Prop :=
  e.cudaBackend=Bool.true ∧ e.currentDevice=device ∧ e.streamDevice=device

def ValidView (device : Nat) (shape : List Nat) (v : View) : Prop :=
  v.tensor.shape=shape ∧ v.tensor.dtype=.fp32 ∧ v.tensor.ordinary=Bool.true ∧
    v.tensor.device=device ∧ v.tensor.strides.length=4 ∧ span v.tensor≤v.tensor.capacity
def ValidDense (device : Nat) (shape : List Nat) (t : Tensor) : Prop :=
  t.shape=shape ∧ t.dtype=.fp32 ∧ t.ordinary=Bool.true ∧ t.device=device ∧
    t.contiguous=Bool.true ∧ DenseLayout t.shape t.strides ∧ t.shape.prod≤t.capacity

structure ForwardContract (c : Invocation) (a : Allocations) (e : LaunchEnvironment) : Prop where
  dimensions : Contract c.dims
  launch : LaunchContract c.query.tensor.device e
  query : ValidView c.query.tensor.device (queryShape c.dims) c.query
  key : ValidView c.query.tensor.device (queryShape c.dims) c.key
  value : ValidView c.query.tensor.device (valueShape c.dims) c.value
  queryDense : ValidDense c.query.tensor.device (queryShape c.dims) a.queryDense
  keyDense : ValidDense c.query.tensor.device (queryShape c.dims) a.keyDense
  valueDense : ValidDense c.query.tensor.device (valueShape c.dims) a.valueDense
  output : ValidDense c.query.tensor.device (valueShape c.dims) a.output
  physical : ForwardPhysicalAllocation c a

structure BackwardContract (c : Invocation) (a : Allocations) (e : LaunchEnvironment) : Prop where
  dimensions : Contract c.dims
  launch : LaunchContract c.query.tensor.device e
  incoming : ValidView c.query.tensor.device (valueShape c.dims) c.incoming
  queryDense : ValidDense c.query.tensor.device (queryShape c.dims) a.queryDense
  keyDense : ValidDense c.query.tensor.device (queryShape c.dims) a.keyDense
  valueDense : ValidDense c.query.tensor.device (valueShape c.dims) a.valueDense
  incomingDense : ValidDense c.query.tensor.device (valueShape c.dims) a.incomingDense
  queryGradient : ValidDense c.query.tensor.device (queryShape c.dims) a.queryGradient
  keyGradient : ValidDense c.query.tensor.device (queryShape c.dims) a.keyGradient
  valueGradient : ValidDense c.query.tensor.device (valueShape c.dims) a.valueGradient
  physical : BackwardPhysicalAllocation c a

structure ForwardResult (α : Type) where
  output : Tensor
  context : Context
  memory : Memory α
  physical : Nat → Cell α
structure BackwardResult (α : Type) where
  gradients : Tensor × Tensor × Tensor
  memory : Memory α
  physical : Nat → Cell α

/-- The public forward executes only its actual grid, returns O, and saves
resolved dense Q/K/V and dimensions for the later backward. -/
def publicForward (M : Algebra α) (D : DotPrimitive α) (c : Invocation) (a : Allocations)
    (copied : Nat → Cell α) (fallback : Memory α) : Option (ForwardResult α) := do
  let result ← executeForward M D c.dims (logical a copied fallback)
  pure ⟨returnedForward a,savedContext c a,result,publishForward a result copied⟩

def withContext (ctx : Context) (a : Allocations) : Allocations :=
  {a with queryDense:=ctx.query,keyDense:=ctx.key,valueDense:=ctx.value}
/-- Backward uses the saved descriptors, then executes intra followed by inter
on current memory after the separately supplied incoming cotangent is copied. -/
def publicBackward (M : Algebra α) (D : DotPrimitive α) (ctx : Context) (a : Allocations)
    (copied : Nat → Cell α) (fallback : Memory α) : Option (BackwardResult α) := do
  let alloc := withContext ctx a
  let result ← executeBackward M D ctx.dims (logical alloc copied fallback)
  pure ⟨returnedBackward alloc,result,publishBackward alloc result copied⟩

theorem with_saved (c : Invocation) (a : Allocations) : withContext (savedContext c a) a=a := by
  cases a; rfl

theorem publicForward_execution (M : Algebra α) (D : DotPrimitive α) (c : Invocation) (a : Allocations)
    (copied : Nat → Cell α) (fallback : Memory α) :
    publicForward M D c a copied fallback=some
      ⟨returnedForward a,savedContext c a,modelForward M D c.dims (logical a copied fallback),
       publishForward a (modelForward M D c.dims (logical a copied fallback)) copied⟩ := by
  simp only [publicForward,forward_execution,bind,Option.bind_some,pure]
theorem publicBackward_execution (M : Algebra α) (D : DotPrimitive α) (c : Invocation) (a : Allocations)
    (copied : Nat → Cell α) (fallback : Memory α) :
    publicBackward M D (savedContext c a) a copied fallback=some
      ⟨returnedBackward a,modelBackward M D c.dims (logical a copied fallback),
       publishBackward a (modelBackward M D c.dims (logical a copied fallback)) copied⟩ := by
  unfold publicBackward
  rw [with_saved]
  simp only [savedContext,backward_execution,bind,Option.bind_some,pure]

/-- Ordinary autograd lifetime/version binding for the saved input values.
Only input coordinates are compared across time; this supplies no output. -/
structure SavedInputs (c : Invocation) (a : Allocations) (forwardCopied backwardCopied : Nat → Cell α) : Prop where
  query : ∀ (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.N) (f : Fin c.dims.D),
    (backwardCopied (a.queryDense.base+packed c.dims c.dims.D b h t f)).read .real=
      (forwardCopied (a.queryDense.base+packed c.dims c.dims.D b h t f)).read .real
  key : ∀ (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.N) (f : Fin c.dims.D),
    (backwardCopied (a.keyDense.base+packed c.dims c.dims.D b h t f)).read .real=
      (forwardCopied (a.keyDense.base+packed c.dims c.dims.D b h t f)).read .real
  value : ∀ (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.N) (f : Fin c.dims.E),
    (backwardCopied (a.valueDense.base+packed c.dims c.dims.E b h t f)).read .real=
      (forwardCopied (a.valueDense.base+packed c.dims c.dims.E b h t f)).read .real

#axiomsClean with_saved
#axiomsClean publicForward_execution
#axiomsClean publicBackward_execution
#specNonCircular ForwardContract avoiding [publicForward,modelForward]
#specNonCircular BackwardContract avoiding [publicBackward,modelBackward]
#specNonCircular SavedInputs avoiding [publicForward,publicBackward,modelForward,modelBackward]
end
end VeriTile.Bench.Repairs.LightningAttention.Host
