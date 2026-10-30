import bench.tritonbench_g.rotary_transform.Proofs.Guards
import bench.tritonbench_g.rotary_emb.Proofs.Binding

namespace VeriTile.Bench.Repairs.RotaryTransform.Binding
open Checked
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
open Classical

abbrev API := VeriTile.Bench.Repairs.SwigluTriton.Memory.Ops

def key (t : Tensor) (index : Nat) : Key := ⟨t.device,t.base+4*index⟩

/-- Ordinary empty_like/inplace metadata. Fresh allocation, shape, identity
and local allocator behavior are the only Tensor-runtime boundary here. -/
structure Output (c : Call) where
  tensor : Tensor
  /-- Actual pointers returned by the original cosine/sine contiguous calls. -/
  coefficientBase : Bool → Nat
  coefficientAligned : c.host.shape.coefficientRows*c.host.shape.half≠0 → ∀ sine,coefficientBase sine%4=0
  allocationStart : tensor.base=tensor.allocationBase
  allocationFits : tensor.base+tensor.allocationBytes≤2^64
  shape : tensor.shape=c.input.shape
  dtype : tensor.dtype="float32"
  unit : tensor.elementBytes=4
  device : tensor.device=c.host.device
  aligned : tensor.shape.prod≠0 → tensor.base%4=0
  contains : 4*Host.size c.host.shape≤tensor.allocationBytes
  canonical : tensor.shape.prod≠0 → tensor.strides=Checked.canonical tensor.shape
  inplace : c.host.shape.inplace=Bool.true → tensor=c.input
  freshIdentity : c.host.shape.inplace=Bool.false → tensor.object=c.host.freshIdentity ∧ tensor.object≠c.host.inputIdentity
  freshAllocation : c.host.shape.inplace=Bool.false → ∀ t∈tensors c,separate tensor t

def dataBound (c : Call) := Host.size c.host.shape+8*c.host.shape.heads*c.host.shape.width+c.host.shape.blockK
def coefficientBound (c : Call) (initial : Physical.Bytes) (batch : Nat) :=
  ((configuration c initial).shift batch+c.host.shape.maximum+8)*c.host.shape.half+c.host.shape.blockK

/-- The actual post-preparation guard; no bound on an unused no-launch
pointer is required. Metadata pointers are contained by tensorCheck. -/
def pointerCheck (c : Call) (out : Output c) (initial : Physical.Bytes) : Bool :=
  Host.empty c.host.shape || decide (
    c.input.base+4*dataBound c≤2^64 ∧ out.tensor.base+4*dataBound c≤2^64 ∧
    ∀ b : Fin c.host.shape.batches,∀ sine : Bool,
      out.coefficientBase sine+4*coefficientBound c initial b.val≤2^64)

theorem pointer_limits (c : Call) (out : Output c) (initial : Physical.Bytes)
    (checked : pointerCheck c out initial=Bool.true) (nonempty : Host.empty c.host.shape=Bool.false)
    (offset : Nat) (inside : offset<dataBound c) :
    c.input.base+4*offset+4≤2^64 ∧ out.tensor.base+4*offset+4≤2^64 := by
  simp only [pointerCheck,nonempty,Bool.false_or,decide_eq_true_eq] at checked
  constructor <;> omega

theorem coefficient_pointer_limits (c : Call) (out : Output c) (initial : Physical.Bytes)
    (checked : pointerCheck c out initial=Bool.true) (nonempty : Host.empty c.host.shape=Bool.false)
    (batch : Fin c.host.shape.batches) (sine : Bool) (offset : Nat)
    (inside : offset<coefficientBound c initial batch.val) :
    out.coefficientBase sine+4*offset+4≤2^64 := by
  simp only [pointerCheck,nonempty,Bool.false_or,decide_eq_true_eq] at checked
  have h := checked.2.2 batch sine
  omega

/-- Direct byte reads through the supplied backing starts. The runtime
cumulative view aliases the original CU bytes; no CU copy is performed. -/
def mirror {M : Type} (api : API M Native.Word Physical.Byte) (c : Call) (out : Output c) (m : M) : Physical.Bytes :=
  fun region b =>
    let tensor := match region with
      | .input => c.input | .output => out.tensor
      | .cosine => c.cosine | .sine => c.sine
      | .cumulative => c.cumulative | .offsets => c.offsets
      | .other 0 => c.cumulative | .other 1 => c.offsets
      | .other 2 => c.cosine | .other 3 => c.sine
      | .other _ => c.input
    api.observe m tensor.device (tensor.base+b)

def sourceResult (c : Call) (callerDevice : Nat) (p : Native.Instructions) (initial : Physical.Bytes) :=
  Public.invoke c.host c.views callerDevice p initial

/-- Finite final stores of the executed source, including the original
out-of-place suffix copy. No target formula or output callback is used. -/
def execute {M : Type} (api : API M Native.Word Physical.Byte) (c : Call) (out : Output c)
    (callerDevice : Nat) (p : Native.Instructions) (m : M) : M :=
  let initial := mirror api c out m
  let result := sourceResult c callerDevice p initial
  stores api (key out.tensor) (Physical.load result.memory c.host.shape.output)
    (List.range (Host.size c.host.shape)) m

theorem indices_apart (t : Tensor) (a b : Nat) (different : a≠b) : (key t a).disjoint (key t b) := by
  simp only [key,Key.disjoint]
  by_cases h : a<b
  · exact Or.inr (Or.inl (by omega))
  · exact Or.inr (Or.inr (by omega))

theorem read_execute {M : Type} (api : API M Native.Word Physical.Byte) (c : Call) (out : Output c)
    (callerDevice : Nat) (p : Native.Instructions) (m : M) (i : Nat) (hi : i<Host.size c.host.shape) :
    api.read (execute api c out callerDevice p m) (key out.tensor i)=
      Physical.load (sourceResult c callerDevice p (mirror api c out m)).memory c.host.shape.output i := by
  apply readback api _ _ _ m i (List.mem_range.mpr hi)
  · intro j _ equal
    have same : j=i := by
      by_contra ne
      exact Key.disjoint_ne (indices_apart out.tensor j i ne) equal
    rw [same]
  · intro j _ different
    exact indices_apart out.tensor i j (by intro same; subst j; exact different rfl)

def Outside (c : Call) (out : Output c) (owner address : Nat) : Prop :=
  ∀ i,i<Host.size c.host.shape → ¬(key out.tensor i).covers owner address

theorem execute_frame {M : Type} (api : API M Native.Word Physical.Byte) (c : Call) (out : Output c)
    (callerDevice : Nat) (p : Native.Instructions) (m : M) (owner address : Nat)
    (outside : Outside c out owner address) :
    api.observe (execute api c out callerDevice p m) owner address=api.observe m owner address := by
  apply bytes_frame
  intro i hi
  exact outside i (List.mem_range.mp hi)

structure Codec {M : Type} (api : API M Native.Word Physical.Byte) : Prop where
  bytes : ∀ m k i,api.observe m k.owner (k.address+i.val)=Physical.encode (api.read m k) i

def expected (c : Call) (p : Native.Instructions) (initial : Physical.Bytes) (b r h d : Nat) : Native.Word :=
  Public.nativeExpected c.host c.views p initial b r h d

theorem values {M : Type} (api : API M Native.Word Physical.Byte) (c : Call) (out : Output c)
    (callerDevice : Nat) (p : Native.Instructions) (m : M)
    (accepted : check c (mirror api c out m)=Bool.true) (b r h d : Nat)
    (hb : b<c.host.shape.batches) (hr : r<(configuration c (mirror api c out m)).length b)
    (hh : h<c.host.shape.heads) (hd : d<c.host.shape.width) :
    api.read (execute api c out callerDevice p m)
      (key out.tensor (c.host.shape.address ((configuration c (mirror api c out m)).start b+r) h d))=
        expected c p (mirror api c out m) b r h d := by
  let initial := mirror api c out m
  have bound := Host.address_bound c.host.shape _ h d
    (row_bound (configuration c initial) (config_valid c initial accepted) b r hb hr) hh hd
  rw [read_execute api c out callerDevice p m _ bound]
  change Physical.load (sourceResult c callerDevice p initial).memory c.host.shape.output
    (c.host.shape.address ((configuration c initial).start b+r) h d)=expected c p initial b r h d
  rw [←Physical.decode_encode (expected c p initial b r h d)]
  unfold Physical.load
  apply congrArg Physical.decode
  funext i
  change (sourceResult c callerDevice p initial).memory c.host.shape.output
    (4*c.host.shape.address ((configuration c initial).start b+r) h d+i.val)=
      Physical.encode (expected c p initial b r h d) i
  have hb' := Public.payload c.host c.views callerDevice p initial
      (config_valid c initial accepted) b r h d hb hr hh hd i
  exact hb'

structure Result (M : Type) where
  tensor : Tensor
  deviceRestored : Nat
  memory : M

def invoke {M : Type} (api : API M Native.Word Physical.Byte) (c : Call) (out : Output c)
    (callerDevice : Nat) (p : Native.Instructions) (m : M) : Option (Result M) :=
  if check c (mirror api c out m) && pointerCheck c out (mirror api c out m) then
    some ⟨out.tensor,callerDevice,execute api c out callerDevice p m⟩ else none

theorem complete {M : Type} (api : API M Native.Word Physical.Byte) (codec : Codec api)
    (c : Call) (out : Output c) (callerDevice : Nat) (p : Native.Instructions) (m : M)
    (accepted : check c (mirror api c out m)=Bool.true)
    (pointers : pointerCheck c out (mirror api c out m)=Bool.true) :
    ∃ result,invoke api c out callerDevice p m=some result ∧ result.tensor=out.tensor ∧
      result.deviceRestored=callerDevice ∧
      (c.host.shape.inplace=Bool.true → result.tensor=c.input) ∧
      (∀ b r h d,b<c.host.shape.batches → r<(configuration c (mirror api c out m)).length b →
        h<c.host.shape.heads → d<c.host.shape.width →
        api.read result.memory (key result.tensor (c.host.shape.address
          ((configuration c (mirror api c out m)).start b+r) h d))=
            expected c p (mirror api c out m) b r h d) ∧
      (∀ b r h d i,b<c.host.shape.batches → r<(configuration c (mirror api c out m)).length b →
        h<c.host.shape.heads → d<c.host.shape.width →
        api.observe result.memory result.tensor.device (result.tensor.base+4*c.host.shape.address
          ((configuration c (mirror api c out m)).start b+r) h d+i.val)=
            Physical.encode (expected c p (mirror api c out m) b r h d) i) ∧
      (∀ owner address,Outside c out owner address →
        api.observe result.memory owner address=api.observe m owner address) := by
  refine ⟨⟨out.tensor,callerDevice,execute api c out callerDevice p m⟩,
    by simp [invoke,accepted,pointers],rfl,rfl,out.inplace,?_,?_,?_⟩
  · exact values api c out callerDevice p m accepted
  · intro b r h d i hb hr hh hd
    change api.observe (execute api c out callerDevice p m)
      (key out.tensor (c.host.shape.address ((configuration c (mirror api c out m)).start b+r) h d)).owner
      ((key out.tensor (c.host.shape.address ((configuration c (mirror api c out m)).start b+r) h d)).address+i.val)=_
    rw [codec.bytes,values api c out callerDevice p m accepted b r h d hb hr hh hd]
  · exact execute_frame api c out callerDevice p m

theorem every_token (c : Call) (initial : Physical.Bytes) (accepted : check c initial=Bool.true)
    (token : Nat) (ht : token<c.host.shape.total) :
    ∃ b r,b<c.host.shape.batches ∧ r<(configuration c initial).length b ∧
      (configuration c initial).start b+r=token := by
  let config := configuration c initial
  have valid := config_valid c initial accepted
  have bound : token<config.start config.shape.batches := by
    rw [start_total config valid]
    exact ht
  obtain ⟨b,hb,lo,hi⟩ := intervals_cover config valid config.shape.batches (by omega) token bound
  refine ⟨b,token-config.start b,hb,?_,?_⟩
  · change token-config.start b<config.length b
    omega
  · change config.start b+(token-config.start b)=token
    omega

#axiomsClean every_token
#axiomsClean pointer_limits
#axiomsClean coefficient_pointer_limits
#axiomsClean read_execute
#axiomsClean values
#axiomsClean complete
#specNonCircular expected avoiding [sourceResult,execute,invoke]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RotaryTransform.Binding
