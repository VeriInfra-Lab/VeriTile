import bench.repairs.tritonbench_g.lightning_attention.Proofs.HostMetadata
import bench.repairs.tritonbench_g.chunk_delta_fwd.HostCall
namespace VeriTile.Bench.Repairs.LightningAttention.Host
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
abbrev Tensor := ChunkDeltaFwd.Host.Tensor
abbrev denseStrides := ChunkDeltaFwd.Host.denseStrides
abbrev span := ChunkDeltaFwd.Host.span
abbrev separate := ChunkDeltaFwd.Host.separate

/-- Tensor storage metadata plus PyTorch's lazy negative-view bit. A logical
read applies the scalar negation operation before a contiguous copy. -/
structure View where
  tensor : Tensor
  lazyNeg : Bool

structure Invocation where
  dims : Dimensions
  (query key value incoming : View)

structure Allocations where
  (queryDense keyDense valueDense incomingDense : Tensor)
  (output queryGradient keyGradient valueGradient : Tensor)

/-- Exactly the three resolved dense forward inputs saved by the wrapper. -/
structure Context where
  dims : Dimensions
  (query key value : Tensor)

def savedContext (c : Invocation) (a : Allocations) : Context :=
  ⟨c.dims,a.queryDense,a.keyDense,a.valueDense⟩
def queryShape (d : Dimensions) := [d.B,d.H,d.N,d.D]
def valueShape (d : Dimensions) := [d.B,d.H,d.N,d.E]
def Invocation.inputs (c : Invocation) : List Tensor :=
  [c.query.tensor,c.key.tensor,c.value.tensor,c.incoming.tensor]
def Allocations.inputs (a : Allocations) : List Tensor :=
  [a.queryDense,a.keyDense,a.valueDense,a.incomingDense]
def Allocations.outputs (a : Allocations) : List Tensor :=
  [a.output,a.queryGradient,a.keyGradient,a.valueGradient]

/-- PyTorch contiguity permits arbitrary strides on singleton axes. This is
the ordinary dense layout rule on the nonempty tensors admitted by Contract. -/
def DenseLayout : List Nat → List Nat → Prop
  | [],[] => True
  | n::tail,stride::strides => (n=1 ∨ stride=tail.prod) ∧ DenseLayout tail strides
  | _,_ => False

/-- Wrapper checks and ordinary view-storage validity. Strides may be zero,
overlapping, or noncontiguous; storage offsets are already included in base. -/
structure InvocationContract (c : Invocation) : Prop where
  dimensions : Contract c.dims
  queryShape : c.query.tensor.shape=Host.queryShape c.dims
  keyShape : c.key.tensor.shape=Host.queryShape c.dims
  valueShape : c.value.tensor.shape=Host.valueShape c.dims
  incomingShape : c.incoming.tensor.shape=Host.valueShape c.dims
  descriptors : ∀ t∈c.inputs, t.dtype=.fp32 ∧ t.ordinary=Bool.true ∧
    t.device=c.query.tensor.device ∧ t.strides.length=4 ∧ span t≤t.capacity

/-- Successful materialization/allocation metadata only. Resolved inputs can
reuse or alias input storage. Output freshness is expressed by PhysicalAllocation. -/
structure AllocationContract (c : Invocation) (a : Allocations) : Prop where
  queryShape : a.queryDense.shape=Host.queryShape c.dims
  keyShape : a.keyDense.shape=Host.queryShape c.dims
  valueShape : a.valueDense.shape=Host.valueShape c.dims
  incomingShape : a.incomingDense.shape=Host.valueShape c.dims
  outputShape : a.output.shape=Host.valueShape c.dims
  queryGradientShape : a.queryGradient.shape=Host.queryShape c.dims
  keyGradientShape : a.keyGradient.shape=Host.queryShape c.dims
  valueGradientShape : a.valueGradient.shape=Host.valueShape c.dims
  descriptors : ∀ t∈a.inputs++a.outputs, t.dtype=.fp32 ∧ t.ordinary=Bool.true ∧
    t.device=c.query.tensor.device ∧ t.contiguous=Bool.true ∧
    DenseLayout t.shape t.strides ∧ t.shape.prod≤t.capacity

def offset4 (src : Tensor) (b h t f : Nat) : Nat :=
  b*src.strides[0]!+h*src.strides[1]!+t*src.strides[2]!+f*src.strides[3]!
def packed (d : Dimensions) (width b h t f : Nat) : Nat :=
  ((b*d.H+h)*d.N+t)*width+f
def callerRead {α : Type} [Inhabited α] (negate : α → α)
    (physical : Nat → Cell α) (src : View) (b h t f : Nat) : α :=
  let raw := (physical (src.tensor.base+offset4 src.tensor b h t f)).read .real
  if src.lazyNeg then negate raw else raw

/-- The local semantic binding for resolve_neg().contiguous(). This says only
what each copied input coordinate contains; no computed output is supplied. -/
structure ResolveCopies {α : Type} [Inhabited α] (negate : α → α)
    (c : Invocation) (a : Allocations) (before after : Nat → Cell α) : Prop where
  query : ∀ (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.N) (f : Fin c.dims.D),
    (after (a.queryDense.base+packed c.dims c.dims.D b h t f)).read .real=
      callerRead negate before c.query b h t f
  key : ∀ (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.N) (f : Fin c.dims.D),
    (after (a.keyDense.base+packed c.dims c.dims.D b h t f)).read .real=
      callerRead negate before c.key b h t f
  value : ∀ (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.N) (f : Fin c.dims.E),
    (after (a.valueDense.base+packed c.dims c.dims.E b h t f)).read .real=
      callerRead negate before c.value b h t f
  incoming : ∀ (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.N) (f : Fin c.dims.E),
    (after (a.incomingDense.base+packed c.dims c.dims.E b h t f)).read .real=
      callerRead negate before c.incoming b h t f

/-- Forward materialization happens before forward allocation/publication. -/
structure ForwardCopies {α : Type} [Inhabited α] (negate : α → α)
    (c : Invocation) (a : Allocations) (before after : Nat → Cell α) : Prop where
  query : ∀ (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.N) (f : Fin c.dims.D),
    (after (a.queryDense.base+packed c.dims c.dims.D b h t f)).read .real=
      callerRead negate before c.query b h t f
  key : ∀ (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.N) (f : Fin c.dims.D),
    (after (a.keyDense.base+packed c.dims c.dims.D b h t f)).read .real=
      callerRead negate before c.key b h t f
  value : ∀ (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.N) (f : Fin c.dims.E),
    (after (a.valueDense.base+packed c.dims c.dims.E b h t f)).read .real=
      callerRead negate before c.value b h t f

/-- Incoming materialization happens later, when backward receives its cotangent.
Its source physical state may contain the already returned forward output. -/
structure IncomingCopy {α : Type} [Inhabited α] (negate : α → α)
    (c : Invocation) (a : Allocations) (before after : Nat → Cell α) : Prop where
  incoming : ∀ (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.N) (f : Fin c.dims.E),
    (after (a.incomingDense.base+packed c.dims c.dims.E b h t f)).read .real=
      callerRead negate before c.incoming b h t f

def descriptor (a : Allocations) (r : RegionName) : Option Tensor :=
  if r="query" then some a.queryDense else if r="key" then some a.keyDense else
  if r="value" then some a.valueDense else if r="incoming" then some a.incomingDense else
  if r="output" then some a.output else if r="queryGradient" then some a.queryGradient else
  if r="keyGradient" then some a.keyGradient else if r="valueGradient" then some a.valueGradient else none
def logical {α : Type} (a : Allocations) (physical : Nat → Cell α)
    (fallback : Memory α) : Memory α := fun r o =>
  match descriptor a r with | some t => physical (t.base+o) | none => fallback r o
def returnedForward (a : Allocations) : Tensor := a.output
def returnedBackward (a : Allocations) : Tensor × Tensor × Tensor :=
  (a.queryGradient,a.keyGradient,a.valueGradient)

theorem saved_dense (c : Invocation) (a : Allocations) :
    (savedContext c a).query=a.queryDense ∧ (savedContext c a).key=a.keyDense ∧
      (savedContext c a).value=a.valueDense := ⟨rfl,rfl,rfl⟩
theorem gradient_routing (a : Allocations) :
    (returnedBackward a).1=a.queryGradient ∧ (returnedBackward a).2.1=a.keyGradient ∧
      (returnedBackward a).2.2=a.valueGradient := ⟨rfl,rfl,rfl⟩

theorem offset4_dense (d : Dimensions) (src : Tensor) (width b h t f : Nat)
    (hs : src.shape=[d.B,d.H,d.N,width]) (hd : src.strides=denseStrides src.shape) :
    offset4 src b h t f=packed d width b h t f := by
  simp [offset4,hd,hs,denseStrides,ChunkDeltaFwd.Host.denseStrides,packed]
  ring

theorem query_copy {α : Type} [Inhabited α] (negate : α → α)
    (c : Invocation) (a : Allocations) (before after : Nat → Cell α) (fallback : Memory α)
    (copies : ResolveCopies negate c a before after)
    (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.N) (f : Fin c.dims.D) :
    ((logical a after fallback) "query" ((b.val*c.dims.H+h.val)*c.dims.N*c.dims.D+
      t.val*c.dims.D+f.val)).read .real=callerRead negate before c.query b h t f := by
  have he : (b.val*c.dims.H+h.val)*c.dims.N*c.dims.D+t.val*c.dims.D+f.val=
      packed c.dims c.dims.D b h t f := by unfold packed; ring
  simpa [logical,descriptor,he] using copies.query b h t f
theorem key_copy {α : Type} [Inhabited α] (negate : α → α)
    (c : Invocation) (a : Allocations) (before after : Nat → Cell α) (fallback : Memory α)
    (copies : ResolveCopies negate c a before after)
    (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.N) (f : Fin c.dims.D) :
    ((logical a after fallback) "key" ((b.val*c.dims.H+h.val)*c.dims.N*c.dims.D+
      t.val*c.dims.D+f.val)).read .real=callerRead negate before c.key b h t f := by
  have he : (b.val*c.dims.H+h.val)*c.dims.N*c.dims.D+t.val*c.dims.D+f.val=
      packed c.dims c.dims.D b h t f := by unfold packed; ring
  simpa [logical,descriptor,he] using copies.key b h t f
theorem value_copy {α : Type} [Inhabited α] (negate : α → α)
    (c : Invocation) (a : Allocations) (before after : Nat → Cell α) (fallback : Memory α)
    (copies : ResolveCopies negate c a before after)
    (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.N) (f : Fin c.dims.E) :
    ((logical a after fallback) "value" ((b.val*c.dims.H+h.val)*c.dims.N*c.dims.E+
      t.val*c.dims.E+f.val)).read .real=callerRead negate before c.value b h t f := by
  have he : (b.val*c.dims.H+h.val)*c.dims.N*c.dims.E+t.val*c.dims.E+f.val=
      packed c.dims c.dims.E b h t f := by unfold packed; ring
  simpa [logical,descriptor,he] using copies.value b h t f
theorem incoming_copy {α : Type} [Inhabited α] (negate : α → α)
    (c : Invocation) (a : Allocations) (before after : Nat → Cell α) (fallback : Memory α)
    (copies : ResolveCopies negate c a before after)
    (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.N) (f : Fin c.dims.E) :
    ((logical a after fallback) "incoming" ((b.val*c.dims.H+h.val)*c.dims.N*c.dims.E+
      t.val*c.dims.E+f.val)).read .real=callerRead negate before c.incoming b h t f := by
  have he : (b.val*c.dims.H+h.val)*c.dims.N*c.dims.E+t.val*c.dims.E+f.val=
      packed c.dims c.dims.E b h t f := by unfold packed; ring
  simpa [logical,descriptor,he] using copies.incoming b h t f

theorem forward_query_copy {α : Type} [Inhabited α] (negate : α → α)
    (c : Invocation) (a : Allocations) (before after : Nat → Cell α) (fallback : Memory α)
    (copies : ForwardCopies negate c a before after)
    (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.N) (f : Fin c.dims.D) :
    ((logical a after fallback) "query" ((b.val*c.dims.H+h.val)*c.dims.N*c.dims.D+
      t.val*c.dims.D+f.val)).read .real=callerRead negate before c.query b h t f := by
  have he : (b.val*c.dims.H+h.val)*c.dims.N*c.dims.D+t.val*c.dims.D+f.val=
      packed c.dims c.dims.D b h t f := by unfold packed; ring
  simpa [logical,descriptor,he] using copies.query b h t f
theorem forward_key_copy {α : Type} [Inhabited α] (negate : α → α)
    (c : Invocation) (a : Allocations) (before after : Nat → Cell α) (fallback : Memory α)
    (copies : ForwardCopies negate c a before after)
    (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.N) (f : Fin c.dims.D) :
    ((logical a after fallback) "key" ((b.val*c.dims.H+h.val)*c.dims.N*c.dims.D+
      t.val*c.dims.D+f.val)).read .real=callerRead negate before c.key b h t f := by
  have he : (b.val*c.dims.H+h.val)*c.dims.N*c.dims.D+t.val*c.dims.D+f.val=
      packed c.dims c.dims.D b h t f := by unfold packed; ring
  simpa [logical,descriptor,he] using copies.key b h t f
theorem forward_value_copy {α : Type} [Inhabited α] (negate : α → α)
    (c : Invocation) (a : Allocations) (before after : Nat → Cell α) (fallback : Memory α)
    (copies : ForwardCopies negate c a before after)
    (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.N) (f : Fin c.dims.E) :
    ((logical a after fallback) "value" ((b.val*c.dims.H+h.val)*c.dims.N*c.dims.E+
      t.val*c.dims.E+f.val)).read .real=callerRead negate before c.value b h t f := by
  have he : (b.val*c.dims.H+h.val)*c.dims.N*c.dims.E+t.val*c.dims.E+f.val=
      packed c.dims c.dims.E b h t f := by unfold packed; ring
  simpa [logical,descriptor,he] using copies.value b h t f
theorem backward_incoming_copy {α : Type} [Inhabited α] (negate : α → α)
    (c : Invocation) (a : Allocations) (before after : Nat → Cell α) (fallback : Memory α)
    (copies : IncomingCopy negate c a before after)
    (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.N) (f : Fin c.dims.E) :
    ((logical a after fallback) "incoming" ((b.val*c.dims.H+h.val)*c.dims.N*c.dims.E+
      t.val*c.dims.E+f.val)).read .real=callerRead negate before c.incoming b h t f := by
  have he : (b.val*c.dims.H+h.val)*c.dims.N*c.dims.E+t.val*c.dims.E+f.val=
      packed c.dims c.dims.E b h t f := by unfold packed; ring
  simpa [logical,descriptor,he] using copies.incoming b h t f

#axiomsClean saved_dense
#axiomsClean gradient_routing
#axiomsClean offset4_dense
#axiomsClean query_copy
#axiomsClean key_copy
#axiomsClean value_copy
#axiomsClean incoming_copy
#axiomsClean forward_query_copy
#axiomsClean forward_key_copy
#axiomsClean forward_value_copy
#axiomsClean backward_incoming_copy
#specNonCircular ForwardCopies avoiding [fwdMemory,fwdOutput,fwdCarry]
#specNonCircular IncomingCopy avoiding [fwdMemory,fwdOutput,fwdCarry]
#specNonCircular ResolveCopies avoiding [fwdMemory,fwdOutput,fwdCarry]
#specNonCircular InvocationContract avoiding [fwdMemory,fwdOutput,fwdCarry]
#specNonCircular AllocationContract avoiding [fwdMemory,fwdOutput,fwdCarry]
end
end VeriTile.Bench.Repairs.LightningAttention.Host
