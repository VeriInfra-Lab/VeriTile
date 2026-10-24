import bench.repairs.tritonbench_g.fp4_to_bf16.Proofs.Contract
import bench.repairs.tritonbench_g.fp4_to_bf16.Proofs.Bytes

namespace VeriTile.Bench.Repairs.Fp4ToBf16.KernelBinding
open VeriTile.Torch VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16 FPValue Host HostCorrect
open Bytes
noncomputable section
set_option maxRecDepth 20000
set_option maxHeartbeats 1000000

/-- Actual packed uint8 Tensor and backing allocation. Physical indices are
in bytes; metadata, storage offset and data_ptr refer to this same object.
The conditional contiguity law is the ordinary Tensor API contract. -/
structure PackedTensor where
  (object storageObject : Nat)
  metadata : TensorMetadata
  (base offset capacity : Nat)
  physicalIndex : Nat → Nat
  span : base+capacity≤2^64
  inBounds : ∀ i,i<metadata.numel → physicalIndex i<capacity
  pointer : 0<metadata.numel → metadata.dataPtr=BitVec.ofNat 64 (base+offset)
  contiguous : metadata.isContiguous=Bool.true → ∀ i,i<metadata.numel → physicalIndex i=offset+i

def PackedTensor.viewOffset (x : PackedTensor) : Nat := if x.metadata.numel=0 then 0 else x.offset
def PackedTensor.viewBase (x : PackedTensor) : Nat := x.base+x.viewOffset
def PackedTensor.value (x : PackedTensor) (heap : Heap) (i : Nat) : BitVec 8 :=
  let raw := heap (x.base+x.physicalIndex i)
  if x.metadata.isNeg then -raw else raw

theorem PackedTensor.window (x : PackedTensor) (h : Checked.InputContract x.metadata) :
    x.viewOffset+x.metadata.numel≤x.capacity := by
  by_cases empty : x.metadata.numel=0
  · simp [PackedTensor.viewOffset,empty]
  · have last := x.inBounds (x.metadata.numel-1) (by omega)
    rw [x.contiguous h.contiguous _ (by omega)] at last
    simp only [PackedTensor.viewOffset,empty,if_false]
    omega

theorem PackedTensor.direct (x : PackedTensor) (h : Checked.InputContract x.metadata)
    (heap : Heap) (i : Nat) (inside : i<x.metadata.numel) :
    heap (x.viewBase+i)=x.value heap i := by
  simp only [PackedTensor.value,h.noNeg,Bool.false_eq_true,if_false,
    x.contiguous h.contiguous i inside,PackedTensor.viewBase,PackedTensor.viewOffset,
    if_neg (by omega : x.metadata.numel≠0),Nat.add_assoc]

theorem PackedTensor.actual_pointer (x : PackedTensor) (h : Checked.InputContract x.metadata)
    (nonempty : 0<x.metadata.numel) : x.metadata.dataPtr.toNat=x.viewBase := by
  rw [x.pointer nonempty,BitVec.toNat_ofNat]
  have window := x.window h
  have span := x.span
  simp only [PackedTensor.viewOffset,if_neg (by omega : x.metadata.numel≠0)] at window
  simp only [PackedTensor.viewBase,PackedTensor.viewOffset,if_neg (by omega : x.metadata.numel≠0)]
  exact Nat.mod_eq_of_lt (by omega)

/-- Actual BF16 output Tensor. Capacity/offset/physicalIndex count two-byte
elements; the signed host storage-size bound is an allocation API observation.
Natural alignment concerns the allocator-created output, not an input guard. -/
structure Bf16Tensor where
  (object storageObject : Nat)
  metadata : TensorMetadata
  (base offset capacity : Nat)
  physicalIndex : Nat → Nat
  dtype : metadata.dtype=.floating .bf16
  layout : metadata.layout=.strided
  span : base+2*capacity≤2^64
  hostCapacity : 2*capacity<2^63
  inBounds : ∀ i,i<metadata.numel → physicalIndex i<capacity
  pointer : 0<metadata.numel → metadata.dataPtr=BitVec.ofNat 64 (base+2*offset)
  aligned : 0<metadata.numel → metadata.dataPtr.toNat%2=0
  contiguous : metadata.isContiguous=Bool.true → ∀ i,i<metadata.numel → physicalIndex i=offset+i

def Bf16Tensor.viewOffset (z : Bf16Tensor) : Nat := if z.metadata.numel=0 then 0 else z.offset
def Bf16Tensor.viewBase (z : Bf16Tensor) : Nat := z.base+2*z.viewOffset
def Bf16Tensor.viewBytes (z : Bf16Tensor) : Nat := 2*(z.capacity-z.viewOffset)
def Bf16Tensor.value (z : Bf16Tensor) (heap : Heap) (i : Nat) : FP.Value .bf16 :=
  let raw : FP.Value .bf16 := ⟨read heap (z.base+2*z.physicalIndex i)⟩
  if z.metadata.isNeg then raw.neg else raw

theorem Bf16Tensor.window (z : Bf16Tensor) (contiguous : z.metadata.isContiguous=Bool.true) :
    z.viewOffset+z.metadata.numel≤z.capacity := by
  by_cases empty : z.metadata.numel=0
  · simp [Bf16Tensor.viewOffset,empty]
  · have last := z.inBounds (z.metadata.numel-1) (by omega)
    rw [z.contiguous contiguous _ (by omega)] at last
    simp only [Bf16Tensor.viewOffset,empty,if_false]
    omega

theorem Bf16Tensor.direct (z : Bf16Tensor) (contiguous : z.metadata.isContiguous=Bool.true)
    (noNeg : z.metadata.isNeg=Bool.false) (heap : Heap) (i : Nat) (inside : i<z.metadata.numel) :
    (z.value heap i).bits=read heap (z.viewBase+2*i) := by
  simp only [Bf16Tensor.value,noNeg,Bool.false_eq_true,if_false,
    z.contiguous contiguous i inside,Bf16Tensor.viewBase,Bf16Tensor.viewOffset,
    if_neg (by omega : z.metadata.numel≠0),Nat.mul_add,Nat.add_assoc]

theorem Bf16Tensor.actual_pointer (z : Bf16Tensor)
    (contiguous : z.metadata.isContiguous=Bool.true) (nonempty : 0<z.metadata.numel) :
    z.metadata.dataPtr.toNat=z.viewBase := by
  rw [z.pointer nonempty,BitVec.toNat_ofNat]
  have window := z.window contiguous
  have span := z.span
  simp only [Bf16Tensor.viewOffset,if_neg (by omega : z.metadata.numel≠0)] at window
  simp only [Bf16Tensor.viewBase,Bf16Tensor.viewOffset,if_neg (by omega : z.metadata.numel≠0)]
  exact Nat.mod_eq_of_lt (by omega)

/-- Exact shape/device identities and fresh backing storage for the actual
selected launch. No result values or kernel execution occur in these facts. -/
structure Binding (h : Input) (x : PackedTensor) (z : Bf16Tensor) : Prop where
  inputShape : x.metadata.shape=h.shape.input
  outputShape : z.metadata.shape=h.shape.output
  device : z.metadata.device=x.metadata.device
  contiguous : z.metadata.isContiguous=Bool.true
  noNeg : z.metadata.isNeg=Bool.false
  regions : h.input≠h.output
  separate : x.base+x.capacity≤z.base ∨ z.base+2*z.capacity≤x.base

theorem Binding.input_count (b : Binding h x z) : x.metadata.numel=h.shape.count := by
  rw [TensorMetadata.numel,b.inputShape,input_numel]
theorem Binding.output_count (b : Binding h x z) : z.metadata.numel=2*h.shape.count := by
  rw [TensorMetadata.numel,b.outputShape,output_numel]
  omega

/-- Derive the existing finite source certificate from these actual Tensor
allocations; no additional count cutoff is imposed on the Python input. -/
def Binding.storage (b : Binding h x z) (checked : Checked.InputContract x.metadata) :
    Finite.Storage h.shape.count where
  inputBase := x.base
  outputBase := z.viewBase
  inputOffset := x.viewOffset
  inputCapacity := x.capacity
  outputCapacity := z.viewBytes
  inputFits := by have := x.window checked; rw [b.input_count] at this; exact this
  outputFits := by
    have := z.window b.contiguous
    rw [b.output_count] at this
    unfold Bf16Tensor.viewBytes
    omega
  hostCapacity := by
    have := z.hostCapacity
    unfold Bf16Tensor.viewBytes
    omega
  inputAddress := x.span
  outputAddress := by
    have := z.window b.contiguous
    have := z.span
    unfold Bf16Tensor.viewBase Bf16Tensor.viewBytes
    omega
  separate := by
    have := z.window b.contiguous
    have sep := b.separate
    unfold Bf16Tensor.viewBase Bf16Tensor.viewBytes
    rcases sep with sep | sep
    · left; omega
    · right; omega

theorem Binding.input_pointer (b : Binding h x z) (checked : Checked.InputContract x.metadata)
    (nonempty : 0<x.metadata.numel) :
    x.metadata.dataPtr.toNat=(b.storage checked).inputPointer := x.actual_pointer checked nonempty
theorem Binding.output_pointer (b : Binding h x z) (checked : Checked.InputContract x.metadata)
    (nonempty : 0<z.metadata.numel) :
    z.metadata.dataPtr.toNat=(b.storage checked).outputBase := z.actual_pointer b.contiguous nonempty

/-- The independent logical high-first packing convention. -/
def logicalNibble (logical : Nat → BitVec 8) (j : Nat) : Bits.Nibble :=
  if j%2=0 then Bits.high (logical (j/2)) else Bits.low (logical (j/2))

theorem logical_nibble (b : Binding h x z) (checked : Checked.InputContract x.metadata)
    (heap : Heap) (j : Nat) (inside : j<2*x.metadata.numel) :
    targetNibble (b.storage checked) heap j=logicalNibble (x.value heap) j := by
  have direct := x.direct checked heap (j/2) (by omega)
  simpa only [targetNibble,logicalNibble,Binding.storage,Finite.Storage.inputPointer,
    PackedTensor.viewBase,direct] using
    congrArg (fun raw : BitVec 8 => if j%2=0 then Bits.high raw else Bits.low raw) direct

theorem output_read (b : Binding h x z) (checked : Checked.InputContract x.metadata)
    (heap : Heap) (memory : RegionName → Nat → Cell Word)
    (j : Nat) (inside : j<2*h.shape.count) :
    (z.value (physical h (b.storage checked) heap memory) j).bits=
      (readBf16 (memory h.output j)).bits := by
  rw [z.direct b.contiguous b.noNeg _ _ (by rw [b.output_count]; exact inside)]
  exact physical_read h (b.storage checked) heap memory j inside

theorem input_storage_frame (b : Binding h x z) (checked : Checked.InputContract x.metadata)
    (heap : Heap) (memory : RegionName → Nat → Cell Word) (o : Nat) (ho : o<x.capacity) :
    physical h (b.storage checked) heap memory (x.base+o)=heap (x.base+o) := by
  apply physical_frame
  have sep := (b.storage checked).separate
  have fit := (b.storage checked).outputFits
  change x.base+x.capacity≤(b.storage checked).outputBase ∨
    (b.storage checked).outputBase+(b.storage checked).outputCapacity≤x.base at sep
  rcases sep with sep | sep <;> omega

#axiomsClean PackedTensor.direct
#axiomsClean PackedTensor.actual_pointer
#axiomsClean Bf16Tensor.direct
#axiomsClean Bf16Tensor.actual_pointer
#axiomsClean Binding.input_pointer
#axiomsClean Binding.output_pointer
#axiomsClean logical_nibble
#axiomsClean output_read
#axiomsClean input_storage_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.Fp4ToBf16.KernelBinding
