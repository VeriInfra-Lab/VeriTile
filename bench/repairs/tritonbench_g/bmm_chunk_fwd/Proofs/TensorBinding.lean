import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.FP
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.BmmChunkFwd.TensorBinding
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open DirectFP
noncomputable section

/-- The optional sequence object uses ordinary INT64 Tensor/storage API facts.
Its payload is read as eight physical bytes and decoded as a signed 64-bit ID. -/
structure Int64Tensor where
  object : Nat
  metadata : VeriTile.Torch.TensorMetadata
  storageObject : Nat
  storageBase : Nat
  storageElements : Nat
  storageOffset : Nat
  physicalIndex : Nat → Nat

structure Int64Laws (t : Int64Tensor) : Prop where
  dtype : t.metadata.dtype=.other "int64"
  layout : t.metadata.layout=.strided
  span : t.storageBase+8*t.storageElements≤2^64
  bounds : ∀ i,i<t.metadata.numel → t.physicalIndex i<t.storageElements
  pointer : 0<t.metadata.numel →
    t.metadata.dataPtr=BitVec.ofNat 64 (t.storageBase+8*t.storageOffset)
  contiguous : t.metadata.isContiguous=Bool.true → ∀ i,i<t.metadata.numel →
    t.physicalIndex i=t.storageOffset+i

/-- Binding of actual objects to the exact metadata queried by admission.
No successful kernel call, native output, or arithmetic law is a field. -/
structure Inputs (p : Metadata) (allocation : Direct.Allocation p) where
  a : VeriTile.Torch.FP32.Tensor
  b : VeriTile.Torch.FP32.Tensor
  seq : Int64Tensor
  aLaws : VeriTile.Torch.FP32.TensorLaws a
  bLaws : VeriTile.Torch.FP32.TensorLaws b
  seqLaws : p.hasSeq=Bool.true → Int64Laws seq
  aMetadata : a.metadata=p.tensor .a
  bMetadata : b.metadata=p.tensor .b
  seqMetadata : p.hasSeq=Bool.true → seq.metadata=p.tensor .seq
  aObject : a.object=allocation.inputObjects .a
  bObject : b.object=allocation.inputObjects .b
  seqObject : p.hasSeq=Bool.true → seq.object=allocation.inputObjects .seq

/-- The supplied output is an already allocated Tensor object. This connects
the kernel result to its Tensor storage; it does not implement allocation. -/
structure Output (p : Metadata) (allocation : Direct.Allocation p) where
  tensor : VeriTile.Torch.FP32.Tensor
  laws : VeriTile.Torch.FP32.TensorLaws tensor
  object : tensor.object=allocation.object
  shape : tensor.metadata.shape=(shape p).outputShape
  device : tensor.metadata.device=(p.tensor .a).device
  contiguous : tensor.metadata.isContiguous=Bool.true
  noNeg : tensor.metadata.isNeg=Bool.false
  pointer : tensor.metadata.dataPtr.toNat=allocation.pointer

abbrev readWord (heap : Heap) (address : Nat) : VeriTile.Torch.FP32.Word := ⟨load32 heap address 0⟩

theorem physical_address (t : VeriTile.Torch.FP32.Tensor)
    (laws : VeriTile.Torch.FP32.TensorLaws t) (contiguous : t.metadata.isContiguous=Bool.true)
    (i : Nat) (bound : i<t.metadata.numel) :
    t.metadata.dataPtr.toNat+4*i=t.storage.address (t.physicalIndex i) := by
  have positive : 0<t.metadata.numel := Nat.zero_lt_of_lt bound
  have cap := VeriTile.Torch.FP32.contiguous_capacity t laws contiguous positive
  have span := t.storage.spanFits
  have fit : t.storage.address t.storageOffset<2^64 := by
    simp only [VeriTile.Torch.FP32.Storage.address]
    omega
  rw [show t.metadata.dataPtr=t.dataPtr from rfl,laws.pointer positive,
    BitVec.toNat_ofNat,Nat.mod_eq_of_lt fit,laws.contiguous contiguous i bound]
  simp only [VeriTile.Torch.FP32.Storage.address]
  omega

theorem direct_word (t : VeriTile.Torch.FP32.Tensor)
    (laws : VeriTile.Torch.FP32.TensorLaws t) (contiguous : t.metadata.isContiguous=Bool.true)
    (noNeg : t.metadata.isNeg=Bool.false) (heap : Heap) (i : Nat) (bound : i<t.metadata.numel) :
    ⟨load32 heap t.metadata.dataPtr.toNat i⟩ =
      VeriTile.Torch.FP32.logicalValue (readWord heap) t laws i := by
  simp only [VeriTile.Torch.FP32.logicalValue,noNeg,Bool.false_eq_true,↓reduceIte,readWord]
  have addr := physical_address t laws contiguous i bound
  apply congrArg (fun w => (⟨w⟩ : VeriTile.Torch.FP32.Word))
  unfold load32
  apply congrArg (BitVec.ofNat 32)
  apply congrArg List.sum
  apply List.map_congr_left
  intro b _
  simp only [Nat.mul_zero,Nat.add_zero,addr]

theorem a_load (p : Metadata) (h : InputContract p) (allocation : Direct.Allocation p)
    (t : Inputs p allocation) (heap : Heap) (i : Nat) (bound : i<(p.tensor .a).numel) :
    ⟨load32 heap (base p .a) i⟩=
      VeriTile.Torch.FP32.logicalValue (readWord heap) t.a t.aLaws i := by
  have hc := h.contiguous .a (Or.inl (by decide))
  have hn := h.noNeg .a (Or.inl (by decide))
  simpa only [base,t.aMetadata] using direct_word t.a t.aLaws
    (by simpa only [t.aMetadata] using hc) (by simpa only [t.aMetadata] using hn)
    heap i (by simpa only [t.aMetadata] using bound)

theorem b_load (p : Metadata) (h : InputContract p) (allocation : Direct.Allocation p)
    (t : Inputs p allocation) (heap : Heap) (i : Nat) (bound : i<(p.tensor .b).numel) :
    ⟨load32 heap (base p .b) i⟩=
      VeriTile.Torch.FP32.logicalValue (readWord heap) t.b t.bLaws i := by
  have hc := h.contiguous .b (Or.inl (by decide))
  have hn := h.noNeg .b (Or.inl (by decide))
  simpa only [base,t.bMetadata] using direct_word t.b t.bLaws
    (by simpa only [t.bMetadata] using hc) (by simpa only [t.bMetadata] using hn)
    heap i (by simpa only [t.bMetadata] using bound)

theorem seq_address (t : Int64Tensor) (laws : Int64Laws t)
    (contiguous : t.metadata.isContiguous=Bool.true) (i : Nat) (bound : i<t.metadata.numel) :
    t.metadata.dataPtr.toNat+8*i=t.storageBase+8*t.physicalIndex i := by
  have positive : 0<t.metadata.numel := Nat.zero_lt_of_lt bound
  have address := laws.bounds i bound
  rw [laws.contiguous contiguous i bound] at address
  have span := laws.span
  have fit : t.storageBase+8*t.storageOffset<2^64 := by omega
  rw [laws.pointer positive,BitVec.toNat_ofNat,Nat.mod_eq_of_lt fit,
    laws.contiguous contiguous i bound]
  omega

theorem seq_load (p : Metadata) (h : InputContract p) (allocation : Direct.Allocation p)
    (t : Inputs p allocation) (heap : Heap) (hasSeq : p.hasSeq=Bool.true)
    (i : Nat) (bound : i<(p.tensor .seq).numel) :
    (load64 heap (base p .seq) i).toInt=
      (load64 heap t.seq.storageBase (t.seq.physicalIndex i)).toInt := by
  have hc := h.contiguous .seq (Or.inr hasSeq)
  have addr := seq_address t.seq (t.seqLaws hasSeq)
    (by simpa only [t.seqMetadata hasSeq] using hc) i
    (by simpa only [t.seqMetadata hasSeq] using bound)
  rw [t.seqMetadata hasSeq] at addr
  change base p .seq+8*i=t.seq.storageBase+8*t.seq.physicalIndex i at addr
  apply congrArg BitVec.toInt
  unfold load64
  apply congrArg (BitVec.ofNat 64)
  apply congrArg List.sum
  apply List.map_congr_left
  intro b _
  rw [addr]

theorem output_size (p : Metadata) (allocation : Direct.Allocation p)
    (t : Output p allocation) : t.tensor.metadata.numel=(shape p).outputSize := by
  simp only [VeriTile.Torch.TensorMetadata.numel,t.shape,Host.Shape.outputShape,
    Host.Shape.outputSize,Host.Shape.groups]
  cases grouped : (shape p).grouped <;> simp [grouped] <;> ring

theorem output_address (p : Metadata) (allocation : Direct.Allocation p)
    (t : Output p allocation) (i : Nat) (bound : i<(shape p).outputSize) :
    allocation.pointer+4*i=t.tensor.storage.address (t.tensor.physicalIndex i) := by
  rw [←t.pointer]
  exact physical_address t.tensor t.laws t.contiguous i
    (by simpa only [output_size p allocation t] using bound)

/-- Ideal interpretation of the original Tensor payloads. The projection is a
choice of Real carrier semantics, not a claim that TF32 arithmetic is exact. -/
def idealInputs (p : Metadata) (allocation : Direct.Allocation p) (t : Inputs p allocation)
    (heap : Heap) (project : VeriTile.Torch.FP32.Word → ℝ) : Mathematical.Inputs where
  a := fun i => project (VeriTile.Torch.FP32.logicalValue (readWord heap) t.a t.aLaws i)
  b := fun i => project (VeriTile.Torch.FP32.logicalValue (readWord heap) t.b t.bLaws i)
  seq := fun i => load64 heap (base p .seq) i
  output := fun i => project ⟨load32 heap allocation.pointer i⟩

#axiomsClean physical_address
#axiomsClean direct_word
#axiomsClean a_load
#axiomsClean b_load
#axiomsClean seq_address
#axiomsClean seq_load
#axiomsClean output_address
end
end VeriTile.Bench.Repairs.BmmChunkFwd.TensorBinding
