import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.KernelWhole
import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.FP

namespace VeriTile.Bench.Repairs.QuantizeKvCopy.KernelTensor
open VeriTile.Triton FP.Structural Kernel
noncomputable section
open Classical
local instance : Inhabited Word := ⟨KernelModels.zero⟩
set_option maxHeartbeats 2000000

def dtype : Role → VeriTile.Torch.DType
  | .input => .floating .fp32
  | .destination => .other "int32"
  | .output => .other "int8"
  | .scale => .floating .fp32

/-- Byte storage may be shared by views with different dtypes. Its identity is
independent of Tensor object identity; no allocation separation is assumed. -/
structure Storage where
  object : Nat
  base : Nat
  bytes : Nat
  fits : base+bytes≤2^64

/-- Task-local mixed-dtype Torch binding. Offsets and physical indices count
Tensor elements, while the underlying storage allocation counts bytes. -/
structure Tensor (role : Role) where
  object : Nat
  metadata : VeriTile.Torch.TensorMetadata
  storage : Storage
  storageOffset : Nat
  physicalIndex : Nat → Nat

structure Laws {r : Role} (t : Tensor r) : Prop where
  dtype : t.metadata.dtype=dtype r
  layout : t.metadata.layout=.strided
  bounds : ∀ i,i<t.metadata.numel → elementBytes r*(t.physicalIndex i+1)≤t.storage.bytes
  pointer : 0<t.metadata.numel → t.metadata.dataPtr=BitVec.ofNat 64 (t.storage.base+elementBytes r*t.storageOffset)
  contiguous : t.metadata.isContiguous=Bool.true → ∀ i,i<t.metadata.numel → t.physicalIndex i=t.storageOffset+i

abbrev Tensors := (r : Role) → Tensor r
def metadata (t : Tensors) : Inputs := fun r => (t r).metadata

def address {r : Role} (t : Tensor r) (i : Nat) : Nat := t.storage.base+elementBytes r*t.physicalIndex i

theorem width_pos (r : Role) : 0<elementBytes r := by cases r <;> decide

theorem direct_address {r : Role} (t : Tensor r) (laws : Laws t)
    (hc : t.metadata.isContiguous=Bool.true) (i : Nat) (hi : i<t.metadata.numel) :
    address t i=t.metadata.dataPtr.toNat+elementBytes r*i := by
  have bound := laws.bounds i hi
  have fits := t.storage.fits
  have hp := width_pos r
  have hcont := laws.contiguous hc i hi
  have hbase : t.storage.base+elementBytes r*t.storageOffset<2^64 := by
    rw [hcont] at bound
    have hmul : elementBytes r*t.storageOffset<elementBytes r*(t.storageOffset+i+1) :=
      Nat.mul_lt_mul_of_pos_left (by omega) hp
    omega
  rw [laws.pointer (by omega),BitVec.toNat_ofNat,Nat.mod_eq_of_lt hbase]
  simp [address,hcont,Nat.mul_add,Nat.add_assoc]

theorem span_capacity (t : Tensors) (laws : ∀ r,Laws (t r))
    (hc : ∀ r,(metadata t r).isContiguous=Bool.true) (r : Role) :
    base (metadata t) r+elementBytes r*(metadata t r).numel≤2^64 := by
  by_cases hz : (metadata t r).numel=0
  · simpa [hz,base] using Nat.le_of_lt (t r).metadata.dataPtr.isLt
  have hi : (metadata t r).numel-1<(metadata t r).numel := by omega
  have bound := (laws r).bounds _ hi
  have fits := (t r).storage.fits
  have ha := direct_address (t r) (laws r) (hc r) _ hi
  have he : elementBytes r*((t r).physicalIndex ((metadata t r).numel-1)+1)=
      elementBytes r*(t r).physicalIndex ((metadata t r).numel-1)+elementBytes r := by simp [Nat.mul_add]
  have hn : elementBytes r*((metadata t r).numel-1)+elementBytes r=elementBytes r*(metadata t r).numel := by
    have : (metadata t r).numel-1+1=(metadata t r).numel := by omega
    simpa only [Nat.mul_add,Nat.mul_one] using congrArg (fun n => elementBytes r*n) this
  change (t r).storage.base+elementBytes r*(t r).physicalIndex _=(t r).metadata.dataPtr.toNat+elementBytes r*((metadata t r).numel-1) at ha
  change (t r).metadata.dataPtr.toNat+elementBytes r*(metadata t r).numel≤2^64
  omega

def fp32At (heap : Heap) (a : Nat) : Word := decodeWord (fun b => heap (a+b.val))
def int32At (heap : Heap) (a : Nat) : MetadataWord := BitVec.ofNat 32 (loadDest heap a)
def int8At (heap : Heap) (a : Nat) : BitVec 8 := BitVec.ofNat 8 (heap a).val

def logicalFP32 {r : Role} (heap : Heap) (t : Tensor r) (i : Nat) : Word := fp32At heap (address t i)
def logicalInt32 (heap : Heap) (t : Tensor .destination) (i : Nat) : MetadataWord := int32At heap (address t i)
def logicalInt8 (heap : Heap) (t : Tensor .output) (i : Nat) : BitVec 8 := int8At heap (address t i)

def words (t : Tensors) (heap : Heap) : Nat → MetadataWord :=
  fun i => int32At heap (base (metadata t) .destination+4*i)

def values (m : Inputs) (w : Nat → MetadataWord) (heap : Heap) (pid g c : Nat) : Word :=
  fp32At heap (base m .input+4*inputOffset (KernelWhole.parameters m w) pid g c)

/-- Every typed region is initialized from the same physical byte heap,
including initial caller code and scale contents needed for the frame. -/
def initial (m : Inputs) (heap : Heap) : State Word where
  mem := fun r i =>
    if r="K" then .mk .real (fp32At heap (base m .input+4*i))
    else if r="D" then .mk .int (int32At heap (base m .destination+4*i)).toInt
    else if r="O" then .mk .int (int8At heap (base m .output+i)).toInt
    else if r="S" then .mk .real (fp32At heap (base m .scale+4*i))
    else .mk .int 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun d _ _ => defaultValue d

 theorem initial_reads (t : Tensors) (heap : Heap) :
    KernelWhole.Reads (metadata t) (words t heap) (values (metadata t) (words t heap) heap) (initial (metadata t) heap) := by
  constructor
  · intro pid g c _ _ _
    simp [initial,values]
  · intro token _
    simp [initial,words]

 theorem metadata_logical (t : Tensors) (laws : ∀ r,Laws (t r)) (heap : Heap)
    (h : InputContract (metadata t) (words t heap)) (i : Nat) (hi : i<(request (metadata t) (words t heap)).tokens) :
    words t heap i=logicalInt32 heap (t .destination) i := by
  rw [logicalInt32,direct_address (t .destination) (laws _) (h.contiguous _) i hi]
  rfl

 theorem input_bound (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w)
    (pid g c : Nat) (hp : pid<(request m w).programs) (hg : g<(request m w).groups) (hc : c<8) :
    inputOffset (KernelWhole.parameters m w) pid g c<(m .input).numel := by
  have hb := RW.linear_bound ((request m w).sourcePrograms*(request m w).groups) 8
    (pid*(request m w).groups+g) c
    (RW.linear_bound (request m w).sourcePrograms (request m w).groups pid g
      (lt_of_lt_of_le hp (checked_domain m w h).program_sources) hg) hc
  have sz := sizes m w h .input
  simp only [elements,elementBytes,groupWidth] at sz
  have he : inputOffset (KernelWhole.parameters m w) pid g c=(pid*(request m w).groups+g)*8+c := by
    simp [inputOffset,KernelWhole.parameters,Nat.add_mul,Nat.mul_assoc]
  rw [he]
  omega

 theorem input_logical (t : Tensors) (laws : ∀ r,Laws (t r)) (heap : Heap)
    (h : InputContract (metadata t) (words t heap)) (pid g c : Nat)
    (hp : pid<(request (metadata t) (words t heap)).programs)
    (hg : g<(request (metadata t) (words t heap)).groups) (hc : c<8) :
    values (metadata t) (words t heap) heap pid g c=
      logicalFP32 heap (t .input) (inputOffset (KernelWhole.parameters (metadata t) (words t heap)) pid g c) := by
  rw [logicalFP32,direct_address (t .input) (laws _) (h.contiguous _) _
    (input_bound _ _ h pid g c hp hg hc)]
  rfl

#axiomsClean direct_address
#axiomsClean span_capacity
#axiomsClean initial_reads
#axiomsClean metadata_logical
#axiomsClean input_logical
#specNonCircular Laws avoiding [Kernel.kernel,FP.Structural.exec,KernelGrid.run]
end
end VeriTile.Bench.Repairs.QuantizeKvCopy.KernelTensor
