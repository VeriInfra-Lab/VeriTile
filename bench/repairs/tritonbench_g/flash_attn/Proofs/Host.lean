import bench.repairs.tritonbench_g.flash_attn.Proofs.Layouts
import bench.repairs.tritonbench_g.flash_attn.Proofs.Physical
import bench.repairs.tritonbench_g.flash_attn.Proofs.RealComplete
-- Preserve the historical Bool spellings after importing the shared Triton DSL.
local notation "true" => Bool.true
local notation "false" => Bool.false
namespace VeriTile.Bench.Repairs.FlashAttn.Host
set_option maxHeartbeats 5000000
structure Tensor where
  shape : Shape
  strides : Strides
  device : Nat
  allocation : Nat
  storageOffset : Nat
  storageBytes : Nat
  dtypeFP32 : Bool
  cuda : Bool
  materialized : Bool
  requiresGrad : Bool
  identity : Nat

/-- Ordinary Tensor metadata only. `axes` is the reverse of the wrapper's
sorted stride scan. The guard neither mentions execution nor output values. -/
structure TensorGuard (t : Tensor) : Prop where
  positive : 0<t.shape.batches ∧ 0<t.shape.heads ∧ 0<t.shape.length ∧ 0<t.shape.dim
  signedDimensions : ∀ i,Layouts.sizes t.shape i<2^31
  positiveStrides : ∀ i,0<Layouts.strides t.strides i ∧ Layouts.strides t.strides i<2^31
  materialized : t.dtypeFP32=true ∧ t.cuda=true ∧ t.materialized=true ∧ t.requiresGrad=false
  span : ∃ axes,axes.Perm [0,1,2,3] ∧ Layouts.Guard (Layouts.sizes t.shape) (Layouts.strides t.strides) axes ∧
    4*(t.storageOffset+Layouts.extent (Layouts.sizes t.shape) (Layouts.strides t.strides) axes)≤t.storageBytes
  formed : envelope t.shape t.strides<2^31

structure Arguments where
  q : Tensor
  k : Tensor
  v : Tensor
  causal : Bool
  scale : Word
  /-- Execution interpretation of the finite internal shared-store winner; not a Python argument. -/
  scratchOwner : Nat → Nat → Nat → Fin 4

structure Contract (a : Arguments) : Prop where
  qguard : TensorGuard a.q
  kguard : TensorGuard a.k
  vguard : TensorGuard a.v
  shapes : a.k.shape=a.q.shape ∧ a.v.shape=a.q.shape
  devices : a.k.device=a.q.device ∧ a.v.device=a.q.device
  allocations : a.q.allocation≠a.k.allocation ∧ a.q.allocation≠a.v.allocation ∧ a.k.allocation≠a.v.allocation
  dimensions : a.q.shape.dim∈[16,32,64,128]
  grid : a.q.shape.batches*a.q.shape.heads≤65535
  scratch : a.q.shape.batches*a.q.shape.heads*a.q.shape.length+127<2^31

def shape (a : Arguments) : Shape := {a.q.shape with causal:=a.causal}
def layout (a : Arguments) (o : Tensor) : Layout := ⟨shape a,a.q.strides,a.k.strides,a.v.strides,o.strides,a.scratchOwner⟩
def grid (a : Arguments) : Nat × Nat × Nat := (ceil a.q.shape.length 128,a.q.shape.batches*a.q.shape.heads,1)
def warps (a : Arguments) := if a.q.shape.dim≤64 then 4 else 8

/-- Local `empty_like`/`empty` facts: shapes, dtype, device, independent live
allocations and concrete storage extents. Contents are unconstrained. -/
structure AllocationFacts (a : Arguments) (o : Tensor) (scratchAllocation : Nat) : Prop where
  outputGuard : TensorGuard o
  outputShape : o.shape=a.q.shape
  outputDevice : o.device=a.q.device
  freshOutput : o.allocation≠a.q.allocation ∧ o.allocation≠a.k.allocation ∧ o.allocation≠a.v.allocation
  freshScratch : scratchAllocation≠a.q.allocation ∧ scratchAllocation≠a.k.allocation ∧
    scratchAllocation≠a.v.allocation ∧ scratchAllocation≠o.allocation

structure Returned where
  tensor : Tensor
  bytes : Physical.Bytes
  restoredDevice : Nat
  launchDevice : Nat
  scratchAllocation : Nat

def invoke (a : Arguments) (o : Tensor) (scratchAllocation callerDevice : Nat)
    (native : Native.Instructions) (schedule : List Work) (m : Physical.Bytes) : Returned :=
  ⟨o,Physical.byteExecute (layout a o) (Native.ops native) a.scale schedule m,
    callerDevice,a.q.device,scratchAllocation⟩

theorem contract_shape (a : Arguments) (h : Contract a) : (shape a).Valid :=
  ⟨h.qguard.positive.1,h.qguard.positive.2.1,h.qguard.positive.2.2.1,h.dimensions⟩

theorem output_layout (a : Arguments) (o : Tensor) (scratch : Nat) (h : AllocationFacts a o scratch) :
    OutputInjective (layout a o) := by
  apply Layouts.output_injective
  obtain ⟨axes,perm,guard,_⟩:=h.outputGuard.span
  refine ⟨axes,perm,?_⟩
  simpa only [layout,shape,Layouts.sizes,h.outputShape] using guard

theorem tensor_access (t : Tensor) (guard : TensorGuard t) (b h n d : Nat)
    (hb : b<t.shape.batches) (hh : h<t.shape.heads) (hn : n<t.shape.length) (hd : d<t.shape.dim) :
    4*(t.storageOffset+address t.strides b h n d+1)≤t.storageBytes := by
  obtain ⟨axes,perm,_,span⟩:=guard.span
  have coords:=Layouts.active_coordinates t.shape b h n d ⟨hb,hh,hn,hd⟩
  have bound:=Layouts.packed_bound (Layouts.sizes t.shape) (Layouts.strides t.strides)
    (Layouts.coords b h n d) axes (fun i _=>coords i)
  rw [Layouts.address_packed _ _ _ _ _ _ perm] at bound
  omega

theorem formed_i32 (t : Tensor) (guard : TensorGuard t) (b h n d : Nat)
    (hb : b<t.shape.batches) (hh : h<t.shape.heads) (hn : n≤t.shape.length+127) (hd : d<t.shape.dim) :
    address t.strides b h n d<2^31 :=
  lt_of_le_of_lt (address_formed_bound t.shape t.strides b h n d hb hh hn hd) guard.formed

theorem invoke_identity (a : Arguments) (o : Tensor) (scratch callerDevice : Nat)
    (native : Native.Instructions) (schedule : List Work) (m : Physical.Bytes) :
    (invoke a o scratch callerDevice native schedule m).tensor=o ∧
    (invoke a o scratch callerDevice native schedule m).restoredDevice=callerDevice ∧
    (invoke a o scratch callerDevice native schedule m).launchDevice=a.q.device := ⟨rfl,rfl,rfl⟩

theorem invoke_output (a : Arguments) (h : Contract a) (o : Tensor) (scratch callerDevice : Nat)
    (allocation : AllocationFacts a o scratch) (actual reference : Native.Instructions)
    (numeric : Native.Correspondence actual reference) (schedule : List Work)
    (complete : Complete (shape a) schedule) (m : Physical.Bytes) (b head row col : Nat)
    (active : (Work.output b head row col).active (shape a)) (i : Fin 4) :
    (invoke a o scratch callerDevice actual schedule m).bytes .o (4*address o.strides b head row col+i.val)=
      Physical.encode (outputValue (layout a o) (Native.ops reference) (Physical.load m) a.scale b head row col) i := by
  exact Physical.whole_bytes (layout a o) (contract_shape a h) (output_layout a o scratch allocation)
    actual reference numeric a.scale schedule complete m (.output b head row col) active i

theorem invoke_frame (a : Arguments) (h : Contract a) (o : Tensor) (scratch callerDevice : Nat)
    (actual : Native.Instructions) (schedule : List Work) (complete : Complete (shape a) schedule)
    (m : Physical.Bytes) (r : Region) (b : Nat) (outside : ¬footprint (layout a o) r (b/4)) :
    (invoke a o scratch callerDevice actual schedule m).bytes r b=m r b :=
  Physical.whole_frame_bytes (layout a o) (contract_shape a h) (Native.ops actual) a.scale schedule complete m r b outside

#axiomsClean output_layout
#axiomsClean tensor_access
#axiomsClean formed_i32
#axiomsClean invoke_identity
#axiomsClean invoke_output
#axiomsClean invoke_frame
#specNonCircular Contract avoiding [invoke,execute,outputValue]
#specNonCircular AllocationFacts avoiding [invoke,execute,outputValue]
end VeriTile.Bench.Repairs.FlashAttn.Host
