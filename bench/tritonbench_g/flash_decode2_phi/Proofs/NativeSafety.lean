import bench.tritonbench_g.flash_decode2_phi.Proofs.TensorBinding
import bench.tritonbench_g.flash_decode2_phi.Proofs.Bounds

namespace VeriTile.Bench.Repairs.FlashDecode2.NativeSafety
open VeriTile.Triton FP.Structural Checked TensorBinding
noncomputable section

structure PointerFacts (o : Observation) : Prop where
  capacity : o.metadata.numel ≤ o.storage.elements
  pointer : 0 < o.metadata.numel → o.metadata.dataPtr = BitVec.ofNat 64 o.storage.base

theorem fp_facts (x : FP32Argument) (contiguous : x.observation.metadata.isContiguous = Bool.true)
    (zero : x.observation.storageOffset = 0) : PointerFacts x.observation := by
  refine ⟨fp_capacity x contiguous zero,?_⟩
  intro positive
  have pointer := x.laws.pointer positive
  change x.tensor.storageOffset = 0 at zero
  simpa [VeriTile.Torch.FP32.Tensor.dataPtr,VeriTile.Torch.FP32.Storage.address,zero]
    using pointer

theorem length_facts (x : LengthArgument)
    (contiguous : x.observation.metadata.isContiguous = Bool.true)
    (zero : x.observation.storageOffset = 0) : PointerFacts x.observation :=
  ⟨length_capacity x contiguous zero,fun positive => by simpa [zero] using x.pointer positive⟩

theorem tensor_facts (c : Call) (t : Tensors c allocations) (h : metadataChecks c) :
    ∀ o ∈ tensors c,PointerFacts o := by
  intro o member
  have contig := (h.1 o member).2.2.2.1
  have zero := (h.1 o member).2.2.2.2.1
  simp only [tensors,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl | rfl | rfl | rfl
  · rw [←t.midObserved] at contig zero ⊢; exact fp_facts t.mid contig zero
  · rw [←t.logObserved] at contig zero ⊢; exact fp_facts t.log contig zero
  · rw [←t.lengthsObserved] at contig zero ⊢; exact length_facts t.lengths contig zero
  · rw [←t.outputObserved] at contig zero ⊢; exact fp_facts t.output contig zero

/-- All active words are inside the actual storage. Formed masked pointers are
also uint64-exact; no bounds/alignment premise is needed for an empty launch. -/
structure Access (c : Call) (o : Observation) : Prop where
  words : ∀ i,i < o.metadata.numel →
    (BitVec.ofNat 64 (o.metadata.dataPtr.toNat+4*i)).toNat = o.storage.base+4*i ∧
    o.storage.base+4*i+4 ≤ o.storage.base+4*o.storage.elements ∧
    o.storage.base+4*i+4 ≤ 2^64 ∧ (o.storage.base+4*i)%4 = 0
  formed : 0 < o.metadata.numel → ∀ i,i < o.metadata.numel+block c →
    (BitVec.ofNat 64 (o.metadata.dataPtr.toNat+4*i)).toNat = o.storage.base+4*i

theorem access (c : Call) (o : Observation) (facts : PointerFacts o)
    (checks : 0 < o.metadata.numel → o.metadata.dataPtr.toNat%4 = 0 ∧
      o.metadata.dataPtr.toNat+4*(o.metadata.numel+block c) ≤ 2^64) : Access c o := by
  have pointer (positive : 0 < o.metadata.numel) : o.metadata.dataPtr.toNat = o.storage.base := by
    have cap := facts.capacity
    have span := o.storage.spanFits
    have bound : o.storage.base < 2^64 := by omega
    simpa only [BitVec.toNat_ofNat,Nat.mod_eq_of_lt bound] using congrArg BitVec.toNat (facts.pointer positive)
  constructor
  · intro i hi
    have positive : 0 < o.metadata.numel := by omega
    have checked := checks positive
    rw [pointer positive] at checked ⊢
    have cap := facts.capacity
    have span := o.storage.spanFits
    rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt (by omega : o.storage.base+4*i < 2^64)]
    exact ⟨rfl,by omega,by omega,by simpa [Nat.add_mod] using checked.1⟩
  · intro positive i hi
    have checked := checks positive
    rw [pointer positive] at checked ⊢
    exact Nat.mod_eq_of_lt (by omega)

def AllAccess (c : Call) : Prop := Checked.active c → ∀ o ∈ tensors c,Access c o

theorem checked_access (c : Call) (t : Tensors c allocations) (memory : Physical.Bytes)
    (h : InputContract c (lengthValues t memory)) : AllAccess c := by
  intro active o member
  exact access c o (tensor_facts c t h.metadata o member) ((h.launch active).1 o member)

theorem fp_coordinates (x : FP32Argument)
    (contiguous : x.observation.metadata.isContiguous = Bool.true)
    (zero : x.observation.storageOffset = 0) (i : Nat) (hi : i < x.observation.metadata.numel) :
    x.tensor.physicalIndex i = i := by
  have index := x.laws.contiguous contiguous i hi
  change x.tensor.storageOffset = 0 at zero
  simpa [zero] using index

structure Coordinates (c : Call) (t : Tensors c allocations) : Prop where
  mid : ∀ i,i < c.mid.metadata.numel → t.mid.tensor.physicalIndex i = i
  log : ∀ i,i < c.log.metadata.numel → t.log.tensor.physicalIndex i = i
  output : ∀ i,i < c.output.metadata.numel → t.output.tensor.physicalIndex i = i
  lengths : ∀ i,i < c.lengths.metadata.numel → t.lengths.physicalIndex i = i

theorem checked_coordinates (c : Call) (t : Tensors c allocations) (h : metadataChecks c) :
    Coordinates c t := by
  have mid := h.1 c.mid (by simp [tensors])
  have log := h.1 c.log (by simp [tensors])
  have out := h.1 c.output (by simp [tensors])
  have len := h.1 c.lengths (by simp [tensors])
  rw [←t.midObserved] at mid
  rw [←t.logObserved] at log
  rw [←t.outputObserved] at out
  rw [←t.lengthsObserved] at len
  constructor
  · intro i hi
    rw [←t.midObserved] at hi
    exact fp_coordinates t.mid mid.2.2.2.1 mid.2.2.2.2.1 i hi
  · intro i hi
    rw [←t.logObserved] at hi
    exact fp_coordinates t.log log.2.2.2.1 log.2.2.2.2.1 i hi
  · intro i hi
    rw [←t.outputObserved] at hi
    exact fp_coordinates t.output out.2.2.2.1 out.2.2.2.2.1 i hi
  · intro i hi
    rw [←t.lengthsObserved] at hi
    have index := t.lengths.contiguous len.2.2.2.1 i hi
    simpa [len.2.2.2.2.1] using index

def Arithmetic (c : Call) (t : Tensors c allocations) (memory : Physical.Bytes) : Prop :=
  ∀ row,row < (geometry c).rows →
    let s := Grid.setup (batches c) (heads c) row (Native.entry t.names (Physical.load memory))
    0 < count s c.blockSeq ∧ count s c.blockSeq ≤ capacity c ∧
    (BitVec.ofNat 64 ((lengthValues t memory (row/heads c)).toNat+c.blockSeq-1)).toInt =
      Int.ofNat ((lengthValues t memory (row/heads c)).toNat+c.blockSeq-1) ∧
    ((Physical.load memory t.names.lengths (row/heads c)).signExtend 64).toInt =
      lengthValues t memory (row/heads c) ∧
    ∀ j,j < count s c.blockSeq → ∀ k : TileIndex [block c],
      j < capacity c ∧ 4*eoff s (heads c) (capacity c) j < 2^63 ∧
      4*voff s (heads c) (capacity c) (width c) j k < 2^63 ∧
      4*ooff s (heads c) (width c) k < 2^63

theorem checked_arithmetic (c : Call) (t : Tensors c allocations) (memory : Physical.Bytes)
    (h : InputContract c (lengthValues t memory)) : Arithmetic c t memory := by
  intro row hr
  let call := nativePreparation c t memory h
  have iterations := Host.row_count (request c h) call row hr
  obtain ⟨positiveHeads,batchIndex,headIndex⟩ := Host.row_indices (request c h) row hr
  have length := call.lengthValues (row/heads c) batchIndex positiveHeads
  refine ⟨iterations.1,iterations.2,?_,?_,?_⟩
  · exact Bounds.signed64 _ (length_add_fits _ _ length.2.2 (request c h).valid.2.2.2.2.1)
  · exact BitVec.toInt_signExtend_of_le (by decide)
  intro j hj k
  obtain ⟨batch,head,block,log,mid,out,masked,logBytes,midBytes,outBytes⟩ :=
    Bounds.source_addresses (request c h) call row j hr hj k
  exact ⟨block,logBytes,midBytes,outBytes⟩

#axiomsClean checked_coordinates
#axiomsClean checked_arithmetic

#axiomsClean fp_facts
#axiomsClean tensor_facts
#axiomsClean access
#axiomsClean checked_access
#axiomsClean fp_coordinates
end
end VeriTile.Bench.Repairs.FlashDecode2.NativeSafety
