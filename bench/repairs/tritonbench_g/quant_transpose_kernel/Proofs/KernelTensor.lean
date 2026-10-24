import bench.repairs.tritonbench_g.quant_transpose_kernel.Proofs.KernelInputs
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.QuantTransposeKernel.KernelTensor
open VeriTile.Triton FP.Structural KernelModels KernelInputs
abbrev Tensor := VeriTile.Torch.FP32.Tensor
abbrev TensorLaws := VeriTile.Torch.FP32.TensorLaws
noncomputable section
set_option maxHeartbeats 1000000

/-- Ordinary strided Tensor API and initial-memory decoding. `matrixIndex`
records the actual stride observations absent from shared TensorMetadata.
It permits transposes, slices, broadcasts and storage offsets; no contiguous
input, finite-value, arithmetic-result or execution premise is present.
API: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.stride.html
and the data_ptr/storage_offset/logical-value references in Torch/FP32.lean. -/
structure InputBinding (md : Metadata) (t : Tensor) (read : Nat → Word)
    (memory : QuantTransposeFinite.Memory) (x : String)
    (physical : QuantTransposeFinite.PhysicalLayout) : Prop where
  shape : t.metadata.shape = md.input.shape
  storageBase : md.input.storageBase = t.storage.base
  storageOffset : md.input.storageOffset = t.storageOffset
  storageLength : md.input.storageLength = t.storage.elements
  cuda : md.input.isCuda = t.metadata.isCuda
  device : md.input.deviceIndex = t.metadata.device.index
  dtype : md.fp32 = t.metadata.isFP32
  resolved : md.resolved = !t.metadata.isNeg
  pointer : md.dataPointer = t.dataPtr.toNat
  matrixIndex : ∀ r c, r < md.input.M → c < md.input.N →
    t.physicalIndex (r * md.input.N + c) =
      t.storageOffset + r * md.input.sam + c * md.input.san
  decode : ∀ o, t.storageOffset + o < t.storage.elements →
    memory x o = some (.mk .fp32 (read (t.storage.address (t.storageOffset + o))).bits)
  bytes : ∀ i, i < t.storage.elements → ∀ k, k < 4 →
    physical.initialBytes (t.storage.address i + k) =
      some (QuantTransposeFinite.byteOf (d := .fp32) (read (t.storage.address i)).bits k)

def values (md : Metadata) (t : Tensor) (laws : TensorLaws t)
    (read : Nat → Word) (r c : Nat) : Word :=
  VeriTile.Torch.FP32.logicalValue read t laws (r * md.input.N + c)

theorem noNeg (md : Metadata) (h : InputContract md) (t : Tensor) (read : Nat → Word)
    (memory : QuantTransposeFinite.Memory) (x : String)
    (physical : QuantTransposeFinite.PhysicalLayout) (b : InputBinding md t read memory x physical) :
    t.metadata.isNeg = Bool.false := by
  have hm := h.metadata
  simp only [metadataChecks, decide_eq_true_eq] at hm
  have hr := hm.2.1
  rw [b.resolved] at hr
  cases hn : t.metadata.isNeg <;> simp_all

theorem matrix_numel (a : QuantTransposeFinite.Input) (rank : a.shape.length = 2) :
    a.numel = a.M * a.N := by
  obtain ⟨m, n, shape⟩ := List.length_eq_two.mp rank
  simp [QuantTransposeFinite.Input.numel, QuantTransposeFinite.Input.M,
    QuantTransposeFinite.Input.N, shape]

theorem input_mem (md : Metadata) (h : InputContract md) (t : Tensor)
    (laws : TensorLaws t) (read : Nat → Word) (memory : QuantTransposeFinite.Memory)
    (x : String) (physical : QuantTransposeFinite.PhysicalLayout)
    (b : InputBinding md t read memory x physical) (r c : Nat)
    (hr : r < md.input.M) (hc : c < md.input.N) :
    memory x (r * md.input.sam + c * md.input.san) =
      some (.mk .fp32 (values md t laws read r c).bits) := by
  have hm := h.metadata
  simp only [metadataChecks, decide_eq_true_eq] at hm
  have count : t.metadata.numel = md.input.M * md.input.N := by
    rw [VeriTile.Torch.TensorMetadata.numel, b.shape]
    exact matrix_numel md.input hm.2.2.2.2.1
  have index : r * md.input.N + c < t.metadata.numel := by
    rw [count]
    simpa [Nat.mul_comm] using
      QuantTransposeFinite.active_encode_bound md.input.N md.input.M c r hc hr
  have bound := laws.inBounds _ index
  rw [b.matrixIndex r c hr hc] at bound
  have decoded := b.decode (r * md.input.sam + c * md.input.san) (by omega)
  simpa [values, VeriTile.Torch.FP32.logicalValue,
    noNeg md h t read memory x physical b, b.matrixIndex r c hr hc, Nat.add_assoc] using decoded

/-- Descriptor words are derived from actual Tensor logical reads, in the
transposed output's flat order. This definition contains no kernel result. -/
def words (md : Metadata) (t : Tensor) (laws : TensorLaws t)
    (read : Nat → Word) (i : Nat) : QuantTransposeFinite.Word .fp32 :=
  (values md t laws read (i % md.input.M) (i / md.input.M)).bits

theorem domain_strided (md : Metadata) (h : InputContract md) (t : Tensor)
    (laws : TensorLaws t) (read : Nat → Word) (memory : QuantTransposeFinite.Memory)
    (alloc : QuantTransposeFinite.Allocations) (x : String)
    (physical : QuantTransposeFinite.PhysicalLayout) (b : InputBinding md t read memory x physical)
    (i : Nat) (hi : i < (parameters md.input alloc x).n) :
    memory x (QuantTransposeFinite.inputOffset (parameters md.input alloc x) i) =
      some (.mk .fp32 (words md t laws read i)) := by
  obtain ⟨hr, hc, _⟩ := QuantTransposeFinite.decode_bounds md.input.M md.input.N i hi
  exact input_mem md h t laws read memory x physical b _ _ hr hc

theorem words_eq_logical (md : Metadata) (h : InputContract md) (t : Tensor)
    (laws : TensorLaws t) (read : Nat → Word) (memory : QuantTransposeFinite.Memory)
    (alloc : QuantTransposeFinite.Allocations) (x : String)
    (physical : QuantTransposeFinite.PhysicalLayout) (b : InputBinding md t read memory x physical)
    (logical : List (QuantTransposeFinite.Word .fp32))
    (xs : Nat → QuantTransposeFinite.Word .fp32)
    (domain : QuantTransposeFinite.Domain .fp32 md.input alloc memory x logical xs)
    (r c : Nat) (hr : r < md.input.M) (hc : c < md.input.N) :
    (⟨xs (c * md.input.M + r)⟩ : Word) = values md t laws read r c := by
  have hi := QuantTransposeFinite.active_encode_bound md.input.M md.input.N r c hr hc
  have original := domain.strided (c * md.input.M + r) hi
  have tensor := domain_strided md h t laws read memory alloc x physical b _ hi
  have same : xs (c * md.input.M + r) = words md t laws read (c * md.input.M + r) := by
    have decoded := congrArg (QuantTransposeFinite.readCell .fp32) (original.symm.trans tensor)
    simpa only [QuantTransposeFinite.read_same, Option.some.injEq] using decoded
  have decode := QuantTransposeFinite.encode_decode md.input.M r c hr
  rw [same]
  simp only [words, decode.1, decode.2]

def inverseValue (t : Tensor) (laws : TensorLaws t) (read : Nat → Word) : Word :=
  VeriTile.Torch.FP32.logicalValue read t laws 0

/-- The prepared scalar buffer actually passed at launch. There is no
assertion about how a maximum or reciprocal produced its word. -/
structure InverseBinding (t : Tensor) (read : Nat → Word)
    (alloc : QuantTransposeFinite.Allocations) (physical : QuantTransposeFinite.PhysicalLayout)
    (initial : State Word) : Prop where
  shape : t.metadata.shape = [1]
  contiguous : t.metadata.isContiguous = Bool.true
  noNeg : t.metadata.isNeg = Bool.false
  device : t.metadata.device.kind = "cuda" ∧ t.metadata.device.index = physical.inverse.ownerDevice
  base : physical.inverse.base = t.storage.address t.storageOffset
  decode : initial.mem ⟨alloc.scale⟩ 0 = .mk .real (read (t.storage.address t.storageOffset))
  bytes : ∀ k, k < 4 →
    physical.initialBytes (t.storage.address t.storageOffset + k) =
      some (QuantTransposeFinite.byteOf (d := .fp32) (read (t.storage.address t.storageOffset)).bits k)

theorem inverse_pointer (t : Tensor) (laws : TensorLaws t) (read : Nat → Word)
    (alloc : QuantTransposeFinite.Allocations) (physical : QuantTransposeFinite.PhysicalLayout)
    (initial : State Word) (b : InverseBinding t read alloc physical initial) :
    BitVec.ofNat 64 physical.inverse.base = t.dataPtr := by
  rw [b.base]
  exact (laws.pointer (by simp [VeriTile.Torch.TensorMetadata.numel, b.shape])).symm

theorem inverse_read (t : Tensor) (laws : TensorLaws t) (read : Nat → Word)
    (alloc : QuantTransposeFinite.Allocations) (physical : QuantTransposeFinite.PhysicalLayout)
    (initial : State Word) (b : InverseBinding t read alloc physical initial) :
    inverseValue t laws read = read (t.storage.address t.storageOffset) := by
  have index : 0 < t.metadata.numel := by simp [VeriTile.Torch.TensorMetadata.numel, b.shape]
  simp [inverseValue, VeriTile.Torch.FP32.logicalValue, b.noNeg,
    laws.contiguous b.contiguous 0 index]

theorem native_reads (a : QuantTransposeFinite.Input) (alloc : QuantTransposeFinite.Allocations)
    (memory : QuantTransposeFinite.Memory) (x : String) (physical : QuantTransposeFinite.PhysicalLayout)
    (inverse : Tensor) (laws : TensorLaws inverse) (read : Nat → Word) (initial : State Word)
    (input : ∀ o w, memory x o = some (.mk .fp32 w) → initial.mem ⟨x⟩ o = .mk .real (FP.Value.mk w))
    (b : InverseBinding inverse read alloc physical initial) :
    NativeReads a alloc memory x physical (inverseValue inverse laws read) initial := by
  refine ⟨input, ?_, ?_⟩
  · rw [inverse_read inverse laws read alloc physical initial b]
    exact b.decode
  · intro k hk
    rw [inverse_read inverse laws read alloc physical initial b, b.base]
    exact b.bytes k hk

/-- INT8 Tensor object observations; the shared FP32 Tensor type cannot be
misused for an INT8 allocation. No allocation or final-byteheap execution. -/
structure OutputTensor where
  object : Nat
  metadata : VeriTile.Torch.TensorMetadata
  storageBase : Nat
  storageElements : Nat
  storageOffset : Nat
  physicalIndex : Nat → Nat

structure OutputBinding (md : Metadata) (input : Tensor) (out : OutputTensor)
    (physical : QuantTransposeFinite.PhysicalLayout) : Prop where
  dtype : out.metadata.dtype = .other "int8"
  layout : out.metadata.layout = .strided
  shape : out.metadata.shape = [md.input.N, md.input.M]
  device : out.metadata.device = input.metadata.device
  ownerDevice : physical.output.ownerDevice = out.metadata.device.index
  contiguous : out.metadata.isContiguous = Bool.true
  noNeg : out.metadata.isNeg = Bool.false
  base : physical.output.base = out.storageBase + out.storageOffset
  pointer : out.metadata.dataPtr = BitVec.ofNat 64 physical.output.base
  indices : ∀ i, i < out.metadata.numel → out.physicalIndex i = out.storageOffset + i
  bounds : ∀ i, i < out.metadata.numel → out.physicalIndex i < out.storageElements
  span : out.storageBase + out.storageElements ≤ 2^64

def outputAddress (out : OutputTensor) (i : Nat) : Nat := out.storageBase + out.physicalIndex i

theorem output_address (md : Metadata) (input : Tensor) (out : OutputTensor)
    (physical : QuantTransposeFinite.PhysicalLayout) (b : OutputBinding md input out physical)
    (r c : Nat) (hr : r < md.input.M) (hc : c < md.input.N) :
    outputAddress out (c * md.input.M + r) = physical.output.base + c * md.input.M + r := by
  have hi : c * md.input.M + r < out.metadata.numel := by
    simp only [VeriTile.Torch.TensorMetadata.numel, b.shape, List.prod_cons, List.prod_nil, Nat.mul_one]
    simpa [Nat.mul_comm] using QuantTransposeFinite.active_encode_bound md.input.M md.input.N r c hr hc
  simp [outputAddress, b.indices _ hi, b.base, Nat.add_assoc]

/-- The native input address is the actual shared Tensor's physical index.
No contiguous-layout premise is used. -/
theorem input_address (md : Metadata) (t : Tensor) (read : Nat → Word)
    (memory : QuantTransposeFinite.Memory) (alloc : QuantTransposeFinite.Allocations)
    (x : String) (physical : QuantTransposeFinite.PhysicalLayout)
    (b : InputBinding md t read memory x physical)
    (ordinary : OrdinaryPhysicalBinding md.input alloc memory x physical)
    (r c : Nat) (hr : r < md.input.M) (hc : c < md.input.N) :
    physical.inputPointer.toNat + (r * md.input.sam + c * md.input.san) * 4 =
      t.storage.address (t.physicalIndex (r * md.input.N + c)) := by
  rw [ordinary.inputPointer, ordinary.inputBase, b.storageBase, b.storageOffset,
    b.matrixIndex r c hr hc]
  simp only [VeriTile.Torch.FP32.Storage.address, QuantTransposeFinite.bytes,
    QuantTransposeFinite.Storage.width]
  ring

#axiomsClean input_mem
#axiomsClean domain_strided
#axiomsClean words_eq_logical
#axiomsClean inverse_read
#axiomsClean inverse_pointer
#axiomsClean native_reads
#axiomsClean output_address
#axiomsClean input_address
#specNonCircular InputBinding avoiding [KernelSource.kernel, KernelExecution.exec, KernelGrid.run]
#specNonCircular InverseBinding avoiding [KernelSource.kernel, KernelExecution.exec, KernelGrid.run]
#specNonCircular OutputBinding avoiding [KernelSource.kernel, KernelExecution.exec, KernelGrid.run]
end
end VeriTile.Bench.Repairs.QuantTransposeKernel.KernelTensor
