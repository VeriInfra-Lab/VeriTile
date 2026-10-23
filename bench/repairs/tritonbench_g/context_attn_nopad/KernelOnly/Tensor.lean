import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.CheckedBridge
import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.Launch

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
noncomputable section
set_option maxHeartbeats 3000000

abbrev Byte := BitVec 8
abbrev Word := BitVec 32
abbrev ByteMemory := Nat → Byte

def load32 (heap : ByteMemory) (p : Nat) : Word :=
  BitVec.ofNat 32 ((heap p).toNat+256*(heap (p+1)).toNat+
    65536*(heap (p+2)).toNat+16777216*(heap (p+3)).toNat)

/-- Actual FP32 Tensor plus ordinary stride/conjugate/untyped-storage queries. -/
structure Payload where
  tensor : VeriTile.Torch.FP32.Tensor
  strides : List Nat
  storageBytes : Nat
  conjugate : Bool

structure PayloadLaws (x : Payload) : Prop where
  tensor : VeriTile.Torch.FP32.TensorLaws x.tensor
  backingElements : x.storageBytes/4 = x.tensor.storage.elements
  backingSpan : x.tensor.storage.base+x.storageBytes≤2^64

def Payload.view (x : Payload) : Checked.View :=
  ⟨x.tensor.metadata,x.strides,x.tensor.storage.base,x.storageBytes,x.tensor.storageOffset,x.conjugate⟩

/-- Ordinary int32 Tensor binding for the two metadata vectors. -/
structure IndexTensor where
  object : Nat
  metadata : VeriTile.Torch.TensorMetadata
  strides : List Nat
  (storageBase storageBytes storageOffset : Nat)
  physicalIndex : Nat → Nat
  conjugate : Bool

structure IndexLaws (x : IndexTensor) : Prop where
  dtype : x.metadata.dtype = .other "int32"
  layout : x.metadata.layout = .strided
  inBounds : ∀ i,i < x.metadata.numel → 4*x.physicalIndex i+4≤x.storageBytes
  contiguous : x.metadata.isContiguous=Bool.true →
    ∀ i,i < x.metadata.numel → x.physicalIndex i=x.storageOffset+i
  pointer : 0 < x.metadata.numel →
    x.metadata.dataPtr = BitVec.ofNat 64 (x.storageBase+4*x.storageOffset)
  span : x.storageBase+x.storageBytes≤2^64

def IndexTensor.view (x : IndexTensor) : Checked.View :=
  ⟨x.metadata,x.strides,x.storageBase,x.storageBytes,x.storageOffset,x.conjugate⟩

structure Request where
  call : Call
  (q k v out : Payload)
  (start length : IndexTensor)
  requestedMax : Nat

def regions : Regions := ⟨"Q","K","V","Out","Start","Length"⟩

def views (r : Request) : Checked.Views :=
  ⟨r.q.view,r.k.view,r.v.view,r.out.view,r.start.view,r.length.view⟩

def payloadCount (r : Request) : Nat := r.call.q.rows*r.call.q.heads*r.call.q.dims

def extent (r : Request) (rg : RegionName) : Nat :=
  if rg∈[regions.query,regions.key,regions.value,regions.out] then payloadCount r
  else if rg∈[regions.start,regions.sequence] then r.call.batches else 0

def base (r : Request) (rg : RegionName) : Nat :=
  if rg=regions.query then r.q.tensor.dataPtr.toNat
  else if rg=regions.key then r.k.tensor.dataPtr.toNat
  else if rg=regions.value then r.v.tensor.dataPtr.toNat
  else if rg=regions.out then r.out.tensor.dataPtr.toNat
  else if rg=regions.start then r.start.metadata.dataPtr.toNat
  else if rg=regions.sequence then r.length.metadata.dataPtr.toNat else 0

def startValues (r : Request) (heap : ByteMemory) (b : Nat) : Int :=
  (load32 heap (r.start.metadata.dataPtr.toNat+4*b)).toInt
def lengthValues (r : Request) (heap : ByteMemory) (b : Nat) : Int :=
  (load32 heap (r.length.metadata.dataPtr.toNat+4*b)).toInt

/-- Only actual API bindings, the checked guards and input metadata contents.
No execution, reference result, compiler, allocation or host-success premise. -/
structure Contract (r : Request) (heap : ByteMemory) : Prop where
  qAPI : PayloadLaws r.q
  kAPI : PayloadLaws r.k
  vAPI : PayloadLaws r.v
  outAPI : PayloadLaws r.out
  startAPI : IndexLaws r.start
  lengthAPI : IndexLaws r.length
  checks : Checked.InputContract r.call regions (views r) (extent r) r.requestedMax
    (startValues r heap) (lengthValues r heap)

def logical {α : Type} (read : Nat → α) (x : Payload) (i : Nat) : α :=
  read (x.tensor.storage.address (x.tensor.physicalIndex i))

theorem payload_pointer (x : Payload) (laws : PayloadLaws x)
    (hc : x.tensor.metadata.isContiguous=Bool.true) (i : Nat)
    (hi : i < x.tensor.metadata.numel) :
    x.tensor.storage.address (x.tensor.physicalIndex i)=x.tensor.dataPtr.toNat+4*i := by
  have cap := VeriTile.Torch.FP32.contiguous_capacity x.tensor laws.tensor hc (by omega)
  have bound : x.tensor.storage.address x.tensor.storageOffset < 2^64 := by
    have span := x.tensor.storage.spanFits
    unfold VeriTile.Torch.FP32.Storage.address
    omega
  rw [laws.tensor.contiguous hc i hi,laws.tensor.pointer (by omega)]
  simp only [VeriTile.Torch.FP32.Storage.address] at bound
  simp only [BitVec.toNat_ofNat,VeriTile.Torch.FP32.Storage.address,Nat.mod_eq_of_lt bound]
  omega

theorem logical_direct {α : Type} (read : Nat → α) (x : Payload) (laws : PayloadLaws x)
    (hc : x.tensor.metadata.isContiguous=Bool.true) (i : Nat) (hi : i < x.tensor.metadata.numel) :
    logical read x i=read (x.tensor.dataPtr.toNat+4*i) := by
  rw [logical,payload_pointer x laws hc i hi]

theorem logical_fp32 (read : Nat → VeriTile.Torch.FP32.Word) (x : Payload) (laws : PayloadLaws x)
    (hn : x.tensor.metadata.isNeg=Bool.false) (i : Nat) :
    logical read x i=VeriTile.Torch.FP32.logicalValue read x.tensor laws.tensor i := by
  simp [logical,VeriTile.Torch.FP32.logicalValue,hn]

theorem payload_span (x : Payload) (laws : PayloadLaws x)
    (hc : x.tensor.metadata.isContiguous=Bool.true) (i : Nat)
    (hi : i < x.tensor.metadata.numel) (j : Fin 4) :
    x.tensor.storage.base≤x.tensor.dataPtr.toNat+4*i+j.val ∧
      x.tensor.dataPtr.toNat+4*i+j.val<x.tensor.storage.base+x.storageBytes := by
  rw [←payload_pointer x laws hc i hi]
  have bound := laws.tensor.inBounds i hi
  have bytes := Nat.mod_lt x.storageBytes (by decide : 0 < 4)
  have quotient := Nat.mod_add_div x.storageBytes 4
  rw [←laws.backingElements] at bound
  unfold VeriTile.Torch.FP32.Storage.address
  have := j.isLt
  omega

def metadataState (r : Request) (heap : ByteMemory) : BlockState where
  mem := fun rg off => if rg=regions.start then ⟨.int,startValues r heap off⟩
    else if rg=regions.sequence then ⟨.int,lengthValues r heap off⟩ else .real 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := r.call.grid.toNumPids

def metadata_observation (r : Request) (heap : ByteMemory) :
    Checked.MetadataObservation r.call regions (metadataState r heap)
      (startValues r heap) (lengthValues r heap) where
  startWords := fun b => load32 heap (r.start.metadata.dataPtr.toNat+4*b)
  lengthWords := fun b => load32 heap (r.length.metadata.dataPtr.toNat+4*b)
  startCPU := fun _ _ => rfl
  lengthCPU := fun _ _ => rfl
  startMemory := by intro b hb; rfl
  lengthMemory := by intro b hb; rfl

theorem labels : Checked.Labels regions := by simp [Checked.Labels,regions]

#specNonCircular Contract avoiding [FP.Structural.exec,FP.Structural.run]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly
