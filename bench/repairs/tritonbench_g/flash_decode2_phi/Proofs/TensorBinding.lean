import bench.repairs.tritonbench_g.flash_decode2_phi.Proofs.Admission
import bench.repairs.tritonbench_g.flash_decode2_phi.Proofs.Physical

namespace VeriTile.Bench.Repairs.FlashDecode2.TensorBinding
open VeriTile.Triton FP.Structural Checked
noncomputable section

/-- Actual shared Tensor with ordinary observed stride/lazy-conjugation metadata. -/
structure FP32Argument where
  tensor : VeriTile.Torch.FP32.Tensor
  laws : VeriTile.Torch.FP32.TensorLaws tensor
  strides : List Nat
  isConj : Bool

def FP32Argument.observation (t : FP32Argument) : Observation :=
  ⟨t.tensor.metadata,t.tensor.storage,t.tensor.storageOffset,t.strides,t.tensor.object,t.isConj⟩

/-- INT32 uses the same four-byte allocation unit. Only ordinary pointer,
contiguous coordinate and element-capacity laws are included. -/
structure LengthArgument where
  observation : Observation
  physicalIndex : Nat → Nat
  bounds : ∀ i,i < observation.metadata.numel → physicalIndex i < observation.storage.elements
  contiguous : observation.metadata.isContiguous = Bool.true →
    ∀ i,i < observation.metadata.numel → physicalIndex i = observation.storageOffset+i
  pointer : 0 < observation.metadata.numel → observation.metadata.dataPtr =
    BitVec.ofNat 64 (observation.storage.base+4*observation.storageOffset)

/-- Named allocation memory is indexed by stable storage identity and byte
offset. Each bound entry is the actual Tensor storage, not a logical payload
snapshot. Aliases may use the same entry. -/
structure Tensors (c : Call) (allocations : RegionName → VeriTile.Torch.FP32.Storage) where
  (mid log output : FP32Argument)
  lengths : LengthArgument
  midObserved : mid.observation = c.mid
  logObserved : log.observation = c.log
  outputObserved : output.observation = c.output
  lengthsObserved : lengths.observation = c.lengths
  names : Native.Bindings
  midStorage : allocations names.mid = mid.tensor.storage
  logStorage : allocations names.log = log.tensor.storage
  outputStorage : allocations names.out = output.tensor.storage
  lengthsStorage : allocations names.lengths = lengths.observation.storage
  allocationIdentity : ∀ r s,(allocations r).object = (allocations s).object → r = s
  callerDevice : Nat

def lengthValues (t : Tensors c allocations) (memory : Physical.Bytes) (b : Nat) : Int :=
  (Physical.load memory t.names.lengths b).toInt

def observationHost (t : Observation) (name : RegionName) (dtype : String) : Host.Tensor :=
  ⟨name,t.metadata.shape,dtype,4,t.metadata.device.index,
    t.metadata.isCuda,if t.metadata.layout = .strided then "strided" else "other",
    t.metadata.isContiguous,t.storageOffset,t.metadata.isNeg,t.isConj⟩

theorem host_valid (c : Call) (h : metadataChecks c) (t : Observation)
    (member : t ∈ tensors c) (name : RegionName) (dtype : String) (shape : List Nat)
    (shapeEq : t.metadata.shape = shape) :
    (observationHost t name dtype).Valid shape dtype c.mid.metadata.device.index := by
  obtain ⟨layout,cuda,device,contiguous,offset,negative,conjugate⟩ := h.1 t member
  have ownerCuda : c.mid.metadata.device.kind = "cuda" := device ▸ cuda
  simp [Host.Tensor.Valid,observationHost,shapeEq,VeriTile.Torch.TensorMetadata.isCuda,
    layout,ownerCuda,device,contiguous,offset,negative,conjugate]

theorem fp_capacity (x : FP32Argument) (contiguous : x.observation.metadata.isContiguous = Bool.true)
    (zero : x.observation.storageOffset = 0) :
    x.observation.metadata.numel ≤ x.observation.storage.elements := by
  by_cases positive : 0 < x.observation.metadata.numel
  · have bound := VeriTile.Torch.FP32.contiguous_capacity x.tensor x.laws contiguous positive
    change x.tensor.storageOffset = 0 at zero
    simpa only [zero,Nat.zero_add] using bound
  · omega

theorem length_capacity (x : LengthArgument)
    (contiguous : x.observation.metadata.isContiguous = Bool.true)
    (zero : x.observation.storageOffset = 0) :
    x.observation.metadata.numel ≤ x.observation.storage.elements := by
  by_cases positive : 0 < x.observation.metadata.numel
  · have bound := x.bounds (x.observation.metadata.numel-1) (by omega)
    rw [x.contiguous contiguous _ (by omega),zero] at bound
    omega
  · omega

theorem apart_names (allocations : RegionName → VeriTile.Torch.FP32.Storage)
    (x y : Observation) (rx ry : RegionName)
    (hx : allocations rx = x.storage) (hy : allocations ry = y.storage)
    (capacity : x.metadata.numel ≤ x.storage.elements) (positive : 0 < x.metadata.numel)
    (apart : Checked.apart x y) : rx ≠ ry := by
  intro same
  have storage : x.storage = y.storage := hx.symm.trans (same ▸ hy)
  unfold Checked.apart at apart
  rw [←storage] at apart
  rcases apart with left | right <;> omega

def preparation (c : Call) (t : Tensors c allocations) (memory : Physical.Bytes)
    (h : InputContract c (lengthValues t memory))
    (before : State α) [Inhabited α]
    (lengths : ∀ b,(before.mem "L" b).read .nat = (lengthValues t memory b).toNat) :
    Host.Preparation α (request c h) where
  before := before
  mid := observationHost c.mid t.names.mid "float32"
  log := observationHost c.log t.names.log "float32"
  lengths := observationHost c.lengths t.names.lengths "int32"
  out := observationHost c.output t.names.out "float32"
  owner := c.mid.metadata.device.index
  callerDevice := t.callerDevice
  midValid := host_valid c h.metadata c.mid (by simp [tensors]) _ _ _ h.metadata.2.2.2.1
  logValid := host_valid c h.metadata c.log (by simp [tensors]) _ _ _ h.metadata.2.2.2.2.1
  lengthsValid := host_valid c h.metadata c.lengths (by simp [tensors]) _ _ _ h.metadata.2.2.2.2.2.1
  outValid := host_valid c h.metadata c.output (by simp [tensors]) _ _ _ h.metadata.2.2.2.2.2.2.1
  disjoint := by
    intro activeRows
    have active : Checked.active c := by change 0 < batches c*heads c at activeRows; constructor <;> nlinarith
    have launches := h.launch active
    have outChecks := h.metadata.1 c.output (by simp [tensors])
    have lengthChecks := h.metadata.1 c.lengths (by simp [tensors])
    have outCapacity : c.output.metadata.numel ≤ c.output.storage.elements := by
      rw [←t.outputObserved]
      exact fp_capacity t.output (t.outputObserved ▸ outChecks.2.2.2.1) (t.outputObserved ▸ outChecks.2.2.2.2.1)
    have lenCapacity : c.lengths.metadata.numel ≤ c.lengths.storage.elements := by
      rw [←t.lengthsObserved]
      exact length_capacity t.lengths (t.lengthsObserved ▸ lengthChecks.2.2.2.1)
        (t.lengthsObserved ▸ lengthChecks.2.2.2.2.1)
    have outPositive : 0 < c.output.metadata.numel := by
      simp only [VeriTile.Torch.TensorMetadata.numel,h.metadata.2.2.2.2.2.2.1,List.prod_cons,List.prod_nil,Nat.mul_one]
      exact Nat.mul_pos active.1 (Nat.mul_pos active.2 h.metadata.2.2.2.2.2.2.2.1.1)
    have lenPositive : 0 < c.lengths.metadata.numel := by
      simpa [VeriTile.Torch.TensorMetadata.numel,h.metadata.2.2.2.2.2.1] using active.1
    have outStorage : allocations t.names.out = c.output.storage := by rw [←t.outputObserved]; exact t.outputStorage
    have midStorage : allocations t.names.mid = c.mid.storage := by rw [←t.midObserved]; exact t.midStorage
    have logStorage : allocations t.names.log = c.log.storage := by rw [←t.logObserved]; exact t.logStorage
    have lenStorage : allocations t.names.lengths = c.lengths.storage := by rw [←t.lengthsObserved]; exact t.lengthsStorage
    exact ⟨apart_names allocations _ _ _ _ outStorage midStorage outCapacity outPositive (launches.2.1 _ (by simp)),
      apart_names allocations _ _ _ _ outStorage logStorage outCapacity outPositive (launches.2.1 _ (by simp)),
      apart_names allocations _ _ _ _ outStorage lenStorage outCapacity outPositive (launches.2.1 _ (by simp)),
      apart_names allocations _ _ _ _ lenStorage midStorage lenCapacity lenPositive launches.2.2.1,
      apart_names allocations _ _ _ _ lenStorage logStorage lenCapacity lenPositive launches.2.2.2⟩
  lengthValues := by
    intro b hb positiveHeads
    have active : Checked.active c := ⟨by change b < batches c at hb; omega,positiveHeads⟩
    obtain ⟨pos,capacity,range⟩ := h.contents active b (List.mem_range.mpr hb)
    rw [lengths]
    change 0 < (lengthValues t memory b).toNat ∧
      (lengthValues t memory b).toNat ≤ Checked.capacity c*c.blockSeq ∧
      (lengthValues t memory b).toNat ≤ 2^31-1
    omega

def nativePreparation (c : Call) (t : Tensors c allocations) (memory : Physical.Bytes)
    (h : InputContract c (lengthValues t memory)) : Host.Preparation Native.Term (request c h) :=
  preparation c t memory h (Native.entry t.names (Physical.load memory)) (fun _ => rfl)

#axiomsClean host_valid
#axiomsClean fp_capacity
#axiomsClean length_capacity
#axiomsClean apart_names
end
end VeriTile.Bench.Repairs.FlashDecode2.TensorBinding
