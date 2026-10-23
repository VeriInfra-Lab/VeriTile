import bench.repairs.tritonbench_g.token_softmax_bloom.Proofs.KernelAddresses

namespace VeriTile.Bench.Repairs.TokenSoftmaxBloom.KernelBinding
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided
open KernelTensor KernelAddresses
noncomputable section
set_option maxHeartbeats 4000000

/-- The raw-word logical coordinate is exactly Torch's logical FP32 read when
the checked negative-view flag is absent. -/
theorem logical_fp32 (read : Nat → Word) (x : Payload)
    (normal : x.tensor.metadata.isNeg=Bool.false) (head token : Nat) :
    logicalValue read x head token=
      (VeriTile.Torch.FP32.logicalValue (fun p => ⟨read p⟩) x.tensor x.laws
        (head*cols x.observation+token)).bits := by
  simp [logicalValue,VeriTile.Torch.FP32.logicalValue,normal]

/-- The admitted signed-int32 load has precisely the same signed int64 value.
This is the cast used by the canonical source before any address arithmetic. -/
theorem metadata_widen (w : Word) (positive : 0 ≤ w.toInt) :
    (w.signExtend 64).toInt=(w.toInt.toNat : Int) := by
  rw [BitVec.toInt_signExtend_of_le (by decide)]
  omega

theorem metadata_bits (w : Word) (positive : 0 ≤ w.toInt) :
    w.signExtend 64=word w.toInt.toNat := by
  apply BitVec.eq_of_toInt_eq
  rw [metadata_widen w positive]
  exact (word_exact _ (by have bound := BitVec.toInt_lt (x := w); omega)).symm

/-- Canonical int32 metadata load at the exact finite-width metadata pointer,
followed by the source's signed extension to int64. -/
def loadedMetadata (read : Nat → Word) (x : MetadataTensor) (b : Nat) : BitVec 64 :=
  (read (word x.metadata.dataPtr.toNat+word 4*(word b*word x.stride)).toNat).signExtend 64

theorem loaded_start (c : TensorCall) (metadataRead : Nat → Word)
    (contract : InputContract c.observation (observedStart c metadataRead) (observedLength c metadataRead))
    (launch : VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked.active c.observation)
    (b : Nat) (hb : b < rows c.start.observation) :
    loadedMetadata metadataRead c.start b=word (start c metadataRead b) := by
  have ptr := (metadata_index c.observation contract.metadata launch b hb c.start.observation
    (by simp [TensorCall.observation])).2
  change (word c.start.metadata.dataPtr.toNat+word 4*(word b*word c.start.stride)).toNat=
    c.start.metadata.dataPtr.toNat+4*(b*c.start.stride) at ptr
  unfold loadedMetadata
  rw [ptr]
  exact metadata_bits _ (KernelAdmission.bounds c metadataRead contract
    ⟨launch.1,launch.2.1,launch.2.2.1⟩ b hb).1

theorem loaded_length (c : TensorCall) (metadataRead : Nat → Word)
    (contract : InputContract c.observation (observedStart c metadataRead) (observedLength c metadataRead))
    (launch : VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked.active c.observation)
    (b : Nat) (hb : b < rows c.start.observation) :
    loadedMetadata metadataRead c.length b=word (length c metadataRead b) := by
  have ptr := (metadata_index c.observation contract.metadata launch b hb c.length.observation
    (by simp [TensorCall.observation])).2
  change (word c.length.metadata.dataPtr.toNat+word 4*(word b*word c.length.stride)).toNat=
    c.length.metadata.dataPtr.toNat+4*(b*c.length.stride) at ptr
  unfold loadedMetadata
  rw [ptr]
  exact metadata_bits _ (KernelAdmission.bounds c metadataRead contract
    ⟨launch.1,launch.2.1,launch.2.2.1⟩ b hb).2.1

theorem native_mask (c : TensorCall) (metadataRead : Nat → Word)
    (contract : InputContract c.observation (observedStart c metadataRead) (observedLength c metadataRead))
    (launch : VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked.active c.observation)
    (b lane : Nat) (hb : b < rows c.start.observation) (hi : lane < (params c).width) :
    (word lane).slt (loadedMetadata metadataRead c.length b)=decide (lane < length c metadataRead b) := by
  rw [loaded_length c metadataRead contract launch b hb]
  exact checked_mask c.observation contract.metadata launch _ _
    (KernelAdmission.bounds c metadataRead contract ⟨launch.1,launch.2.1,launch.2.2.1⟩ b hb).2.2.1 hi

/-- Compose source cell, actual bounded 64-bit native pointer and original
logical Tensor coordinate; the final equality binds the caller storage cell. -/
theorem source_native_load {α} [Inhabited α] (c : TensorCall) (read : Nat → α)
    (metadataRead : Nat → Word) (before : NativeCells α) (binding : InputBinding c read before)
    (contract : InputContract c.observation (observedStart c metadataRead) (observedLength c metadataRead))
    (launch : VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked.active c.observation)
    (b head lane : Nat) (hb : b < rows c.start.observation)
    (hh : head < rows c.input.observation) (hn : lane < length c metadataRead b) :
    let offset := head*c.input.rowStride+(start c metadataRead b+lane)*c.input.colStride
    let pointer := word c.input.tensor.dataPtr.toNat+word 4*
      (word head*word c.input.rowStride+(loadedMetadata metadataRead c.start b+word lane)*word c.input.colStride)
    (view c read metadataRead before "input" offset).read .fp32=read pointer.toNat ∧
      read pointer.toNat=logicalValue read c.input head (start c metadataRead b+lane) ∧
      logicalValue read c.input head (start c metadataRead b+lane)=
        (before (c.input.tensor.storage.address
          (c.input.tensor.physicalIndex (head*cols c.input.observation+start c metadataRead b+lane)))).read .fp32 := by
  have content : contentActive c.observation := ⟨launch.1,launch.2.1,launch.2.2.1⟩
  have bounds := KernelAdmission.bounds c metadataRead contract content b hb
  have cover := KernelAdmission.cover c metadataRead contract content b hb
  have inside : start c metadataRead b+lane < cols c.input.observation := by omega
  have rank : c.input.tensor.metadata.shape.length=2 := contract.metadata.tensors.2.2.2.2.2.2.2.1
  have load := KernelTensor.source_load c read metadataRead before binding rank head (start c metadataRead b+lane) hh inside
  have pointer := (checked_payload_addresses c.observation contract.metadata launch
    head (start c metadataRead b) lane hh (by change start c metadataRead b ≤ cols c.input.observation; omega) (lt_of_lt_of_le hn cover)
    c.input.observation (by simp [TensorCall.observation])).2.1
  change (word c.input.tensor.dataPtr.toNat+word 4*
    (word head*word c.input.rowStride+(word (start c metadataRead b)+word lane)*word c.input.colStride)).toNat=
    c.input.tensor.dataPtr.toNat+4*(head*c.input.rowStride+(start c metadataRead b+lane)*c.input.colStride) at pointer
  dsimp only
  rw [loaded_start c metadataRead contract launch b hb,pointer]
  refine ⟨load.1,load.2.1,?_⟩
  simpa [Nat.add_assoc] using load.2.2

#axiomsClean logical_fp32
#axiomsClean metadata_widen
#axiomsClean native_mask
#axiomsClean source_native_load
end
end VeriTile.Bench.Repairs.TokenSoftmaxBloom.KernelBinding
