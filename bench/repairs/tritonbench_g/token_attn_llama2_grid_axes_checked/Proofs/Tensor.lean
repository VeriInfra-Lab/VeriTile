import VeriTile.Torch.FP32
import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Proofs.Domain

namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked
open VeriTile.Triton FP.Structural
open Whole
noncomputable section

structure Payload where
  tensor : VeriTile.Torch.FP32.Tensor
  laws : VeriTile.Torch.FP32.TensorLaws tensor
  strides : List Nat

/-- Ordinary contiguous int32 Tensor storage laws; values are read from the
current raw-word memory, not supplied as a disconnected metadata array. -/
structure MetadataTensor where
  object : Nat
  metadata : VeriTile.Torch.TensorMetadata
  storage : VeriTile.Torch.FP32.Storage
  storageOffset : Nat
  strides : List Nat
  pointer : 0 < metadata.numel →
    metadata.dataPtr=BitVec.ofNat 64 (storage.address storageOffset)
  bounds : metadata.isContiguous=Bool.true → ∀ i < metadata.numel,
    storageOffset+i < storage.elements

structure Observation where
  metadata : VeriTile.Torch.TensorMetadata
  strides : List Nat

def Payload.observation (t : Payload) : Observation := ⟨t.tensor.metadata,t.strides⟩
def MetadataTensor.observation (t : MetadataTensor) : Observation := ⟨t.metadata,t.strides⟩

structure TensorCall where
  q : Payload
  k : Payload
  output : Payload
  locations : MetadataTensor
  starts : MetadataTensor
  lengths : MetadataTensor
  maximum : Nat

def TensorCall.observations (c : TensorCall) : List Observation :=
  [c.q.observation,c.k.observation,c.output.observation,c.locations.observation,
   c.starts.observation,c.lengths.observation]

def dimensionAt (t : Observation) (i : Nat) := t.metadata.shape[i]?.getD 0
def stride (t : Observation) (i : Nat) := t.strides[i]?.getD 0

def TensorCall.call (t : TensorCall) : Whole.Call where
  batches := dimensionAt t.q.observation 0
  heads := dimensionAt t.q.observation 1
  dim := dimensionAt t.q.observation 2
  kvHeads := dimensionAt t.k.observation 1
  cacheRows := dimensionAt t.k.observation 0
  outputTokens := dimensionAt t.output.observation 1
  maxLength := t.maximum
  qBatch := stride t.q.observation 0
  qHead := stride t.q.observation 1
  qDim := stride t.q.observation 2
  kBatch := stride t.k.observation 0
  kHead := stride t.k.observation 1
  kDim := stride t.k.observation 2
  locBatch := stride t.locations.observation 0
  locSeq := stride t.locations.observation 1
  outHead := stride t.output.observation 0
  outSeq := stride t.output.observation 1
  qSize := t.q.tensor.metadata.numel
  kSize := t.k.tensor.metadata.numel
  locSize := t.locations.metadata.numel
  outSize := t.output.tensor.metadata.numel

abbrev Word := BitVec 32
abbrev Cells (A : Type) := Nat → Cell A

def metadata (t : TensorCall) (read : Nat → Word) : Whole.Metadata t.call where
  lengths := fun i => (read (t.lengths.metadata.dataPtr.toNat+4*i.val)).toInt
  starts := fun i => (read (t.starts.metadata.dataPtr.toNat+4*i.val)).toInt
  locations := fun i => (read (t.locations.metadata.dataPtr.toNat+4*i.val)).toInt

def buffers {A} [Inhabited A] (t : TensorCall) (before : Cells A) : Whole.Buffers A t.call where
  q := fun i => (before (t.q.tensor.dataPtr.toNat+4*i.val)).read .fp32
  k := fun i => (before (t.k.tensor.dataPtr.toNat+4*i.val)).read .fp32
  out := fun i => (before (t.output.tensor.dataPtr.toNat+4*i.val)).read .fp32

def logical {A} [Inhabited A] (before : Cells A) (t : Payload) (i : Nat) : A :=
  (before (t.tensor.storage.address (t.tensor.physicalIndex i))).read .fp32

theorem payload_address (t : Payload) (contiguous : t.tensor.metadata.isContiguous=Bool.true)
    (i : Nat) (hi : i < t.tensor.metadata.numel) :
    t.tensor.dataPtr.toNat+4*i=t.tensor.storage.address (t.tensor.physicalIndex i) := by
  have positive : 0 < t.tensor.metadata.numel := by omega
  have bound := t.laws.inBounds 0 positive
  rw [t.laws.contiguous contiguous 0 positive] at bound
  have span := t.tensor.storage.spanFits
  have base : t.tensor.storage.address t.tensor.storageOffset < 2^64 := by
    simp only [VeriTile.Torch.FP32.Storage.address] at *
    omega
  rw [t.laws.pointer positive]
  change (t.tensor.storage.address t.tensor.storageOffset)%2^64+4*i=_
  rw [Nat.mod_eq_of_lt base,t.laws.contiguous contiguous i hi]
  simp [VeriTile.Torch.FP32.Storage.address,Nat.mul_add,Nat.add_assoc]

theorem payload_read {A} [Inhabited A] (before : Cells A) (t : Payload)
    (contiguous : t.tensor.metadata.isContiguous=Bool.true) (i : Nat)
    (hi : i < t.tensor.metadata.numel) :
    (before (t.tensor.dataPtr.toNat+4*i)).read .fp32=logical before t i := by
  rw [payload_address t contiguous i hi]
  rfl

theorem logical_fp32 (read : Nat → Word) (t : Payload)
    (ordinary : t.tensor.metadata.isNeg=Bool.false) (i : Nat) :
    logical (fun address => Cell.mk .fp32 (read address)) t i=
      (VeriTile.Torch.FP32.logicalValue (fun address => ⟨read address⟩) t.tensor t.laws i).bits := by
  simp [logical,VeriTile.Torch.FP32.logicalValue,ordinary]

#axiomsClean payload_address
#axiomsClean payload_read
#axiomsClean logical_fp32
end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked
