import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.Proofs.Tensor
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Proofs.Binding

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Binding
open VeriTile.Triton
open Native
noncomputable section
abbrev Bytes := Nat → BitVec 8
abbrev byteOf := SoftmaxFlaggemsFiniteSeed.Binding.byteOf

/-- Ordinary FP32 allocation readback at its observed data pointer. Source
stores have already selected and frozen their individual consumer replicas. -/
def callerBytes (Y : RegionName) (y : VeriTile.Torch.TensorMetadata) (before : Bytes)
    (words : RegionName → Nat → Word) (address : Nat) : BitVec 8 :=
  if y.dataPtr.toNat≤address ∧ address<y.dataPtr.toNat+4*y.numel then
    byteOf (words Y ((address-y.dataPtr.toNat)/4)) ((address-y.dataPtr.toNat)%4)
  else before address

theorem coordinate_bytes (Y : RegionName) (y : VeriTile.Torch.TensorMetadata) (before : Bytes)
    (words : RegionName → Nat → Word) (i b : Nat) (hi : i<y.numel) (hb : b<4) :
    callerBytes Y y before words (y.dataPtr.toNat+4*i+b)=byteOf (words Y i) b := by
  have inside : y.dataPtr.toNat≤y.dataPtr.toNat+4*i+b ∧ y.dataPtr.toNat+4*i+b<y.dataPtr.toNat+4*y.numel := by omega
  have index : (y.dataPtr.toNat+4*i+b-y.dataPtr.toNat)/4=i := by omega
  have byte : (y.dataPtr.toNat+4*i+b-y.dataPtr.toNat)%4=b := by omega
  simp only [callerBytes,if_pos inside,index,byte]

theorem exterior_bytes (Y : RegionName) (y : VeriTile.Torch.TensorMetadata) (before : Bytes)
    (words : RegionName → Nat → Word) (address : Nat)
    (outside : address<y.dataPtr.toNat ∨ y.dataPtr.toNat+4*y.numel≤address) :
    callerBytes Y y before words address=before address := by
  have miss : ¬(y.dataPtr.toNat≤address ∧ address<y.dataPtr.toNat+4*y.numel) := by omega
  simp only [callerBytes,if_neg miss]

theorem output_bytes (Y : RegionName) (y : VeriTile.Torch.TensorMetadata) (before : Bytes)
    (count : Nat) (shapeCount : y.numel=count) (expected : Nat → Word)
    (memory : RegionName → Nat → Word) (result : Tensor.Result)
    (output : Tensor.Output Y count expected memory result) :
    (∀ i b, i<count → b<4 → callerBytes Y y before result.words (y.dataPtr.toNat+4*i+b)=byteOf (expected i) b) ∧
    (∀ a, a<y.dataPtr.toNat ∨ y.dataPtr.toNat+4*y.numel≤a → callerBytes Y y before result.words a=before a) := by
  refine ⟨?_,exterior_bytes Y y before result.words⟩
  intro i b hi hb
  rw [coordinate_bytes Y y before result.words i b (by omega) hb,output Y i,if_pos ⟨rfl,hi⟩]

/-- Ordinary observed full storage extent, including a Tensor view's prefix
and suffix. This descriptor is an allocation observation, not a value premise. -/
structure Backing where
  base : Nat
  bytes : Nat

def Within (t : VeriTile.Torch.TensorMetadata) (b : Backing) : Prop :=
  b.base≤t.dataPtr.toNat ∧ t.dataPtr.toNat+4*t.numel≤b.base+b.bytes ∧ b.base+b.bytes≤2^64

def Separate (a b : Backing) : Prop := a.base+a.bytes≤b.base ∨ b.base+b.bytes≤a.base

def wordAt (memory : Bytes) (address : Nat) : Word :=
  BitVec.ofNat 32 ((memory address).toNat + 256*(memory (address+1)).toNat +
    65536*(memory (address+2)).toNat + 16777216*(memory (address+3)).toNat)

/-- Actual little-endian caller bytes at the Tensor's data_ptr determine every
input word. Aliased read tensors share this same before-memory. -/
def InputBytes (region : RegionName) (t : VeriTile.Torch.TensorMetadata)
    (memory : RegionName → Nat → Word) (before : Bytes) : Prop :=
  ∀ i, i<t.numel → memory region i=wordAt before (t.dataPtr.toNat+4*i)

def InputLoads (R : Reductions) (region : RegionName) (t : VeriTile.Torch.TensorMetadata)
    (memory : RegionName → Nat → Word) (before : Bytes) : Prop :=
  ∀ i, i<t.numel → ∀ consumer : Consumer,
    (algebra R).fp32Load (((entry memory).mem region i).read .real) consumer=
      SoftmaxFlaggemsFiniteSeed.Native.Term.load (.leaf region i (wordAt before (t.dataPtr.toNat+4*i)))

theorem input_loads (R : Reductions) (region : RegionName) (t : VeriTile.Torch.TensorMetadata)
    (memory : RegionName → Nat → Word) (before : Bytes) (bound : InputBytes region t memory before) :
    InputLoads R region t memory before := by
  intro i hi consumer
  change SoftmaxFlaggemsFiniteSeed.Native.Term.load (.leaf region i (memory region i))=_
  rw [bound i hi]

/-- Ordinary fresh allocation yields a byte frame over the complete input
storage, even when the input Tensor is a view inside that storage. -/
theorem storage_frame (Y : RegionName) (y : VeriTile.Torch.TensorMetadata) (dst src : Backing)
    (within : Within y dst) (fresh : 0<y.numel → Separate dst src) (before : Bytes)
    (words : RegionName → Nat → Word) (a : Nat) (inside : src.base≤a ∧ a<src.base+src.bytes) :
    callerBytes Y y before words a=before a := by
  apply exterior_bytes
  by_cases hz : y.numel=0
  · omega
  · rcases fresh (by omega) with left|right <;> rcases within with ⟨lo,hi,bound⟩ <;> omega

theorem input_word_frame (Y region : RegionName) (y t : VeriTile.Torch.TensorMetadata) (dst src : Backing)
    (yWithin : Within y dst) (tWithin : Within t src) (fresh : 0<y.numel → Separate dst src)
    (before : Bytes) (words memory : RegionName → Nat → Word) (input : InputBytes region t memory before)
    (i : Nat) (hi : i<t.numel) :
    wordAt (callerBytes Y y before words) (t.dataPtr.toNat+4*i)=memory region i := by
  rw [input i hi]
  have bytes (j : Nat) (hj : j<4) : callerBytes Y y before words (t.dataPtr.toNat+4*i+j)=before (t.dataPtr.toNat+4*i+j) := by
    apply storage_frame Y y dst src yWithin fresh before words
    rcases tWithin with ⟨lo,high,bound⟩
    constructor <;> omega
  simp only [wordAt,bytes 1 (by decide),bytes 2 (by decide),bytes 3 (by decide)]
  have first := bytes 0 (by decide)
  simp only [Nat.add_zero] at first
  rw [first]

#axiomsClean input_word_frame
#axiomsClean input_loads
#axiomsClean storage_frame
#specNonCircular InputBytes avoiding [Tensor.forward,Tensor.backward,FP.Structural.exec]
#specNonCircular Within avoiding [Tensor.forward,Tensor.backward,FP.Structural.exec]

#axiomsClean coordinate_bytes
#axiomsClean exterior_bytes
#axiomsClean output_bytes
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Binding
