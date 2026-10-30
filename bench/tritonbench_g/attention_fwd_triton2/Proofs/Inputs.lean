import bench.tritonbench_g.attention_fwd_triton2.Proofs.Admission
import VeriTile.Torch.Tensor

namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.KernelOnly
open VeriTile Triton FP.Structural Whole001 Source Whole Machine
noncomputable section
set_option maxHeartbeats 4000000

inductive Role where | query | key | value | queryScale | keyScale | output
  deriving DecidableEq, Repr
def reads : List Role := [.query,.key,.value,.queryScale,.keyScale]
def tag : Role → RegionName
  | .query => "Q" | .key => "K" | .value => "V"
  | .queryScale => "QS" | .keyScale => "KS" | .output => "Out"
def byteWidth (r : Role) : Nat := if r=.output then 2 else 4

/-- Original Tensor API observations. The four-dimensional strides are used
for Q/K/V/Out, and packedStrides are the observed three scale strides.
physicalIndex is ordinary Tensor indexing, with d=0 for scale coordinates. -/
structure Tensor where
  object : Nat
  metadata : VeriTile.Torch.TensorMetadata
  strides : Strides
  packedStrides : List Nat
  (ordinary isConj : Bool)
  (storageObject storageBase storageElements storageOffset : Nat)
  physicalIndex : Nat → Nat → Nat → Nat → Nat

structure Call where
  tensors : Role → Tensor
instance : CoeFun Call (fun _=>Role→Tensor) := ⟨Call.tensors⟩
def extent (c : Call) (i : Nat) := (c .query).metadata.shape[i]?.getD 0
def base (c : Call) (r : Role) := (c r).metadata.dataPtr.toNat
def parameters (c : Call) : Parameters where
  Q:=tag .query; K:=tag .key; V:=tag .value
  QS:=tag .queryScale; KS:=tag .keyScale; Out:=tag .output
  q:=(c .query).strides; k:=(c .key).strides; v:=(c .value).strides; o:=(c .output).strides
  batches:=extent c 0; heads:=extent c 1; length:=extent c 2
def pointers (c : Call) : Bases :=
  ⟨base c .query,base c .key,base c .value,base c .queryScale,base c .keyScale,base c .output⟩
def descriptor (c : Call) (r : Role) : Descriptor where
  object:=(c r).object
  device:=(c r).metadata.device.index
  region:=tag r
  dtype:=if (c r).metadata.dtype=.floating .fp32 then .fp32 else .bf16
  shape:=(c r).metadata.shape
  strides:=if r=.queryScale ∨ r=.keyScale then (c r).packedStrides
    else [(c r).strides.batch,(c r).strides.head,(c r).strides.row,(c r).strides.dim]
  negative:=(c r).metadata.isNeg
  conjugate:=(c r).isConj
open Classical in
def arguments (c : Call) : Checked.Arguments (parameters c) where
  query:=descriptor c .query; key:=descriptor c .key; value:=descriptor c .value
  queryScale:=descriptor c .queryScale; keyScale:=descriptor c .keyScale
  cuda:=decide (∀ r∈reads,(c r).metadata.isCuda=Bool.true)
  strided:=decide (∀ r∈reads,(c r).metadata.layout=.strided)
  tensors:=decide (∀ r∈reads,(c r).ordinary=Bool.true)
  strides:=by simp [descriptor,parameters]
  regions:=by simp [descriptor,parameters]

/-- Original admission on actual observations; output allocation properties
are ordinary API laws below, not new guards or successful-run premises. -/
def InputContract (c : Call) : Prop :=
  Checked.inputChecks (parameters c) (arguments c)=Bool.true ∧
  Checked.denseChecks (parameters c)=Bool.true ∧
  Checked.setupChecks (parameters c) (pointers c)=Bool.true
open Classical in
def check (c : Call) : Bool := decide (InputContract c)
theorem accepted_iff (c : Call) : check c=Bool.true ↔ InputContract c := by simp [check]

def rowCount (p : Parameters) (r : Role) : Nat :=
  match r with
  | .queryScale => (p.length+127)/128
  | .keyScale => (p.length+63)/64
  | _ => p.length
def columns (r : Role) : Nat := if r=.queryScale ∨ r=.keyScale then 1 else 96
def shape (p : Parameters) (r : Role) : List Nat :=
  if r=.queryScale ∨ r=.keyScale then [p.batches,p.heads,rowCount p r]
  else [p.batches,p.heads,p.length,96]
def Valid (p : Parameters) (r : Role) (b h n d : Nat) : Prop :=
  b<p.batches ∧ h<p.heads ∧ n<rowCount p r ∧ d<columns r
def wordOffset (c : Call) (r : Role) (b h n d : Nat) : Nat :=
  if r=.queryScale ∨ r=.keyScale then
    b*((c r).packedStrides[0]?.getD 0)+h*((c r).packedStrides[1]?.getD 0)+n*((c r).packedStrides[2]?.getD 0)
  else offset (c r).strides b h n d

/-- Truthful Tensor coordinate/capacity/pointer observations and ordinary
fresh empty_like output facts. There is no output or execution equation. -/
structure TensorLaws (c : Call) : Prop where
  outputShape : (c .output).metadata.shape=shape (parameters c) .output
  outputType : (c .output).metadata.dtype=.floating .bf16
  outputDevice : (c .output).metadata.device=(c .query).metadata.device
  outputLayout : (c .output).metadata.layout=.strided
  outputResolved : (c .output).metadata.isNeg=Bool.false ∧ (c .output).isConj=Bool.false
  outputDense : Live (parameters c) → (c .output).strides=dense (parameters c)
  capacity : ∀ r,(c r).storageBase+byteWidth r*(c r).storageElements≤2^64
  indices : ∀ r b h n d,Valid (parameters c) r b h n d →
    (c r).physicalIndex b h n d=(c r).storageOffset+wordOffset c r b h n d
  inBounds : ∀ r b h n d,Valid (parameters c) r b h n d →
    (c r).physicalIndex b h n d<(c r).storageElements
  pointer : ∀ r,0<(c r).metadata.numel →
    (c r).metadata.dataPtr=BitVec.ofNat 64 ((c r).storageBase+byteWidth r*(c r).storageOffset)
  fresh : ∀ r∈reads,
    (c .output).storageBase+2*(c .output).storageElements≤(c r).storageBase ∨
    (c r).storageBase+4*(c r).storageElements≤(c .output).storageBase

theorem domain (c : Call) (hc : InputContract c) (laws : TensorLaws c) :
    Domain (parameters c) (pointers c) :=
  Checked.admittedDomain _ _ hc.2.1 laws.outputDense hc.2.2

theorem admitted_shapes (c : Call) (hc : InputContract c) (laws : TensorLaws c) (r : Role) :
    (c r).metadata.shape=shape (parameters c) r := by
  have hs:=(Checked.admittedABI _ _ hc.1).shapes
  simp only [Checked.admittedABI,arguments,descriptor] at hs
  rcases hs with ⟨hq,hk,hv,hqs,hks⟩
  cases r <;> simp_all [shape,rowCount,TensorLaws.outputShape]

def logicalAddress (c : Call) (r : Role) (b h n d : Nat) :=
  (c r).storageBase+byteWidth r*(c r).physicalIndex b h n d
def logical {α : Type} (read : Nat→α) (c : Call) (r : Role) (b h n d : Nat) :=
  read (logicalAddress c r b h n d)
def sourceAddress (c : Call) (r : Role) (b h n d : Nat) :=
  base c r+byteWidth r*wordOffset c r b h n d

theorem numel_positive (c : Call) (hc : InputContract c) (laws : TensorLaws c)
    (r : Role) (b h n d : Nat) (valid : Valid (parameters c) r b h n d) :
    0<(c r).metadata.numel := by
  rw [VeriTile.Torch.TensorMetadata.numel,admitted_shapes c hc laws r]
  rcases valid with ⟨hb,hh,hn,hd⟩
  cases r <;> simp_all [shape,rowCount,columns] <;> omega

theorem actual_address (c : Call) (hc : InputContract c) (laws : TensorLaws c)
    (r : Role) (b h n d : Nat) (valid : Valid (parameters c) r b h n d) :
    logicalAddress c r b h n d=sourceAddress c r b h n d := by
  have cap:=laws.capacity r
  have bound:=laws.inBounds r b h n d valid
  rw [laws.indices r b h n d valid] at bound
  have ptr:=laws.pointer r (numel_positive c hc laws r b h n d valid)
  have small : (c r).storageBase+byteWidth r*(c r).storageOffset<2^64 := by
    cases r <;> simp [byteWidth] at * <;> omega
  simp only [logicalAddress,sourceAddress,base,ptr,BitVec.toNat_ofNat,
    Nat.mod_eq_of_lt small,laws.indices r b h n d valid,Nat.mul_add]
  omega

theorem actual_read {α : Type} (read : Nat→α) (c : Call) (hc : InputContract c)
    (laws : TensorLaws c) (r : Role) (b h n d : Nat) (valid : Valid (parameters c) r b h n d) :
    logical read c r b h n d=read (sourceAddress c r b h n d) :=
  congrArg read (actual_address c hc laws r b h n d valid)

theorem scale_offset (c : Call) (hc : InputContract c) (live : Live (parameters c))
    (r : Role) (hr : r=.queryScale ∨ r=.keyScale) (b h n d : Nat) :
    wordOffset c r b h n d=(b*(parameters c).heads+h)*rowCount (parameters c) r+n := by
  have hs:=(Checked.admittedABI _ _ hc.1).scaleStrides live.1 live.2.1 live.2.2
  simp only [Checked.admittedABI,arguments,descriptor] at hs
  rcases hr with rfl|rfl <;> simp_all [wordOffset,rowCount,Nat.add_mul,Nat.mul_assoc]

theorem valid_live (c : Call) (r : Role) (b h n d : Nat)
    (valid : Valid (parameters c) r b h n d) : Live (parameters c) := by
  have hn : 0<(parameters c).length := by
    by_contra h
    have zero : (parameters c).length=0 := by omega
    have bound:=valid.2.2.1
    cases r <;> simp [rowCount,zero] at bound
  exact ⟨by have:=valid.1;omega,by have:=valid.2.1;omega,hn⟩

theorem scale_read {α : Type} (read : Nat→α) (c : Call) (hc : InputContract c)
    (laws : TensorLaws c) (r : Role) (hr : r=.queryScale ∨ r=.keyScale) (b h n d : Nat)
    (valid : Valid (parameters c) r b h n d) :
    logical read c r b h n d=
      read (base c r+4*((b*(parameters c).heads+h)*rowCount (parameters c) r+n)) := by
  rw [actual_read read c hc laws r b h n d valid]
  unfold sourceAddress
  rw [scale_offset c hc (valid_live c r b h n d valid) r hr]
  rcases hr with rfl|rfl <;> rfl

#axiomsClean accepted_iff
#axiomsClean domain
#axiomsClean actual_address
#axiomsClean actual_read
#axiomsClean scale_offset
#axiomsClean scale_read
#specNonCircular InputContract avoiding [FP.Structural.exec,Whole.runPrograms,Whole.answer]
end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.KernelOnly
