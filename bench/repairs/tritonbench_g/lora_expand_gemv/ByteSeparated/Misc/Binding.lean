import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Contract
import VeriTile.Torch.FP32
import bench.repairs.tritonbench_g.swiglu_triton.Proofs.Memory
import bench.repairs.tritonbench_g.adam_update_triton.Proofs.Bytes

namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated
open Contract VeriTile.Triton
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section

namespace Raw
abbrev State := VeriTile.Bench.Repairs.AdamUpdateTriton.Bytes.Memory
abbrev Word := BitVec 32
abbrev read := VeriTile.Bench.Repairs.AdamUpdateTriton.Bytes.read
abbrev unpack := VeriTile.Bench.Repairs.AdamUpdateTriton.Bytes.unpack
abbrev ops := VeriTile.Bench.Repairs.AdamUpdateTriton.Bytes.ops
abbrev byte_of_read := VeriTile.Bench.Repairs.AdamUpdateTriton.Bytes.byte_of_read
end Raw

def tensorKey (t : TensorInfo) (off : Nat) : Key :=
  ⟨t.device.index,t.dataPtr.toNat+4*off⟩

/-- Ordinary byte-storage observations. Only nonempty contiguous windows need
an address bound; a zero-length Tensor may have data_ptr=0 on nonempty storage.
No separation or kernel outcome is a field of this binding. -/
structure Window (t : TensorInfo) (width : Nat) (backing : StorageRange) : Prop where
  fits : backing.base+backing.bytes≤2^64
  bounds : 0<t.numel → backing.base≤t.dataPtr.toNat ∧
    t.dataPtr.toNat+width*t.numel≤backing.base+backing.bytes

theorem Window.active_byte {t : TensorInfo} {width : Nat} {backing : StorageRange}
    (w : Window t width backing) (off byte : Nat)
    (hi : off<t.numel) (hb : byte<width) :
    backing.covers (t.dataPtr.toNat+width*off+byte) ∧
      t.dataPtr.toNat+width*off+byte<2^64 := by
  have h:=w.bounds (by omega)
  have mul:=Nat.mul_le_mul_left width (Nat.succ_le_of_lt hi)
  simp only [Nat.mul_succ] at mul
  constructor
  · unfold StorageRange.covers;omega
  · have fit:=w.fits;omega

/-- Actual core FP32 Tensor objects for X/Y. Storage ranges are the same API
observations used by the new Python guard, including full backing capacity.
Tensor object and StorageImpl identity are distinct and neither implies alias
separation. -/
structure FP32Binding (info : TensorInfo) (backing : StorageRange) where
  tensor : VeriTile.Torch.FP32.Tensor
  laws : VeriTile.Torch.FP32.TensorLaws tensor
  metadata : tensor.metadata=info.toTensorMetadata
  object : tensor.object=info.object
  storageObject : tensor.storage.object=info.backing
  storageOffset : tensor.storageOffset=info.storageOffset
  storageBase : tensor.storage.base=backing.base
  /-- A dtype reinterpretation can leave trailing bytes in untyped storage.
  The complete byte backing, including that tail, remains in the checked span. -/
  storageElements : tensor.storage.elements=backing.bytes/4
  backingFits : backing.base+backing.bytes≤2^64

theorem FP32Binding.pointer_nat (a : FP32Binding info backing)
    (hc : info.isContiguous=Bool.true) (hne : 0< info.numel) :
    info.dataPtr.toNat=a.tensor.storage.base+4*a.tensor.storageOffset := by
  have numel : a.tensor.metadata.numel=info.numel := congrArg VeriTile.Torch.TensorMetadata.numel a.metadata
  have contiguous : a.tensor.metadata.isContiguous=Bool.true := by rw [a.metadata];exact hc
  have cap:=VeriTile.Torch.FP32.contiguous_capacity a.tensor a.laws contiguous (by omega)
  have fit:=a.tensor.storage.spanFits
  have bound : a.tensor.storage.base+4*a.tensor.storageOffset<2^64 := by omega
  have p:=congrArg BitVec.toNat (a.laws.pointer (by omega))
  simpa only [VeriTile.Torch.FP32.Tensor.dataPtr,a.metadata,
    VeriTile.Torch.FP32.Storage.address,BitVec.toNat_ofNat,Nat.mod_eq_of_lt bound] using p

def FP32Binding.window (a : FP32Binding info backing) (hc : info.isContiguous=Bool.true) :
    Window info 4 backing where
  fits := a.backingFits
  bounds := by
    intro nonempty
    have numel : a.tensor.metadata.numel=info.numel := congrArg VeriTile.Torch.TensorMetadata.numel a.metadata
    have contiguous : a.tensor.metadata.isContiguous=Bool.true := by rw [a.metadata];exact hc
    have cap:=VeriTile.Torch.FP32.contiguous_capacity a.tensor a.laws contiguous (by omega)
    have bytes:=a.storageElements
    rw [a.pointer_nat hc nonempty,←a.storageBase]
    omega

def indexBytes (t : TensorInfo) := if t.dtype=.other "int64" then 8 else 4

structure Binding (t : Inputs) (a : Backings) where
  input : FP32Binding t.x a.x
  output : FP32Binding t.y a.y
  weights : Window t.w 2 a.w
  indices : Window t.indices (indexBytes t.indices) a.indices

/-- Raw physical reads agree with the actual nonlazy contiguous FP32 Tensor
logical reads. This law concerns individual inputs, not a kernel result. -/
theorem FP32Binding.logical_read (a : FP32Binding info backing)
    (hc : info.isContiguous=Bool.true) (hn : info.isNeg=Bool.false) (before : Raw.State)
    (off : Nat) (hi : off< info.numel) :
    (VeriTile.Torch.FP32.logicalValue
      (fun address => ⟨Raw.read before ⟨info.device.index,address⟩⟩)
      a.tensor a.laws off).bits=Raw.read before (tensorKey info off) := by
  have numel : a.tensor.metadata.numel=info.numel := congrArg VeriTile.Torch.TensorMetadata.numel a.metadata
  have contiguous : a.tensor.metadata.isContiguous=Bool.true := by rw [a.metadata];exact hc
  have negative : a.tensor.metadata.isNeg=Bool.false := by rw [a.metadata];exact hn
  have physical:=a.laws.contiguous contiguous off (by omega)
  simp only [VeriTile.Torch.FP32.logicalValue,negative,Bool.false_eq_true,↓reduceIte,
    physical,VeriTile.Torch.FP32.Storage.address,tensorKey,
    a.pointer_nat hc (by omega),Nat.mul_add,Nat.add_assoc]

#axiomsClean Window.active_byte
#axiomsClean FP32Binding.pointer_nat
#axiomsClean FP32Binding.logical_read
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated
