import VeriTile.Torch.FP32
import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.Admission

/-! Actual Torch FP32 storage bindings for the existing checked descriptors.
The observations below are ordinary Tensor API facts, not kernel results. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly
open VeriTile Triton
noncomputable section

structure Payload where
  tensor : Torch.FP32.Tensor
  conjugate : Bool

def Payload.descriptor (x : Payload) : Admission.Tensor where
  isTensor := Bool.true
  strided := x.tensor.metadata.isStrided
  fp32 := x.tensor.metadata.isFP32
  cuda := x.tensor.metadata.isCuda
  contiguous := x.tensor.metadata.isContiguous
  negative := x.tensor.metadata.isNeg
  conjugate := x.conjugate
  device := x.tensor.metadata.device.index
  shape := x.tensor.metadata.shape
  byteBase := x.tensor.dataPtr.toNat

/-- The descriptor and its shared-core Tensor refer to the same observed object.
Read aliases and nonzero storage offsets remain permitted. -/
structure Bindings (descriptors : List Admission.Tensor) where
  get : Admission.Tensor → Payload
  observations : ∀ d ∈ descriptors, (get d).descriptor = d
  api : ∀ d ∈ descriptors, Torch.FP32.TensorLaws (get d).tensor

theorem Bindings.shape {ds : List Admission.Tensor} (b : Bindings ds)
    (d : Admission.Tensor) (hd : d ∈ ds) : (b.get d).tensor.metadata.shape = d.shape :=
  congrArg (fun x : Admission.Tensor => x.shape) (b.observations d hd)

theorem Bindings.pointer {ds : List Admission.Tensor} (b : Bindings ds)
    (d : Admission.Tensor) (hd : d ∈ ds) : (b.get d).tensor.dataPtr.toNat = d.byteBase :=
  congrArg (fun x : Admission.Tensor => x.byteBase) (b.observations d hd)

theorem Bindings.contiguous {ds : List Admission.Tensor} (b : Bindings ds)
    (d : Admission.Tensor) (hd : d ∈ ds) (hc : d.contiguous = Bool.true) :
    (b.get d).tensor.metadata.isContiguous = Bool.true :=
  (congrArg (fun x : Admission.Tensor => x.contiguous) (b.observations d hd)).trans hc

theorem Bindings.notNeg {ds : List Admission.Tensor} (b : Bindings ds)
    (d : Admission.Tensor) (hd : d ∈ ds) (hn : d.negative = Bool.false) :
    (b.get d).tensor.metadata.isNeg = Bool.false :=
  (congrArg (fun x : Admission.Tensor => x.negative) (b.observations d hd)).trans hn

/-- Physical address of a valid logical element follows from the actual
storage capacity, pointer observation, and checked contiguity. -/
theorem Bindings.logical_address {ds : List Admission.Tensor} (b : Bindings ds)
    (d : Admission.Tensor) (hd : d ∈ ds) (hc : d.contiguous = Bool.true)
    (i : Nat) (hi : i < d.shape.prod) :
    (b.get d).tensor.storage.address ((b.get d).tensor.physicalIndex i) = d.byteBase + 4*i := by
  let t := (b.get d).tensor
  have laws := b.api d hd
  have bound : i < t.metadata.numel := by
    simpa [t,Torch.TensorMetadata.numel,b.shape d hd] using hi
  have cont := b.contiguous d hd hc
  have capacity := Torch.FP32.contiguous_capacity t laws cont (by omega)
  have span := t.storage.spanFits
  have baseBound : t.storage.address t.storageOffset < 2^64 := by
    unfold Torch.FP32.Storage.address
    omega
  rw [laws.contiguous cont i bound]
  have ptr := b.pointer d hd
  rw [laws.pointer (Nat.zero_lt_of_lt bound)] at ptr
  change (BitVec.ofNat 64 (t.storage.address t.storageOffset)).toNat = d.byteBase at ptr
  rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt baseBound] at ptr
  change t.storage.address (t.storageOffset+i) = d.byteBase+4*i
  simp only [Torch.FP32.Storage.address] at ptr ⊢
  omega

def logical {α : Type} (read : Nat → α) (x : Payload) (i : Nat) : α :=
  read (x.tensor.storage.address (x.tensor.physicalIndex i))

theorem Bindings.logical_direct {α : Type} {ds : List Admission.Tensor}
    (b : Bindings ds) (read : Nat → α) (d : Admission.Tensor) (hd : d ∈ ds)
    (hc : d.contiguous = Bool.true) (i : Nat) (hi : i < d.shape.prod) :
    logical read (b.get d) i = read (d.byteBase + 4*i) := by
  rw [logical,b.logical_address d hd hc i hi]

theorem Bindings.logical_fp32 {ds : List Admission.Tensor} (b : Bindings ds)
    (read : Nat → Torch.FP32.Word) (d : Admission.Tensor) (hd : d ∈ ds)
    (hc : d.contiguous = Bool.true) (hn : d.negative = Bool.false) (i : Nat) (hi : i < d.shape.prod) :
    Torch.FP32.logicalValue read (b.get d).tensor (b.api d hd) i = read (d.byteBase + 4*i) := by
  simp only [Torch.FP32.logicalValue,b.notNeg d hd hn,Bool.false_eq_true,↓reduceIte]
  exact b.logical_direct read d hd hc i hi

#axiomsClean Bindings.logical_address
#axiomsClean Bindings.logical_fp32
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly
