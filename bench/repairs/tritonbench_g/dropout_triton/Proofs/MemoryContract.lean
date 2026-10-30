import bench.repairs.tritonbench_g.dropout_triton.Proofs.Contract
import VeriTile.Torch.FP32

/-! Kernel input bindings. The data input uses shared FP32 Tensor semantics.
The int32 mask reuses the same four-byte storage geometry but has its own dtype
and signed-word interpretation; no FP32 law is applied to int32 metadata.
There is no allocator, wrapper execution, or kernel-output premise here. -/
namespace VeriTile.Bench.Repairs.DropoutTriton.MemoryContract
open VeriTile.Triton VeriTile.Torch.FP32

/-- The common Tensor descriptor records metadata, storage and element indices.
The following laws, rather than the descriptor type, establish int32 meaning.
Torch data_ptr, storage_offset, is_contiguous and resolve_neg references are
recorded in the shared `VeriTile/Torch/FP32.lean` binding. -/
structure Int32Laws (keep : Tensor) : Prop where
  dtype : keep.metadata.dtype = .other "int32"
  layout : keep.metadata.layout = .strided
  inBounds : ∀ i, i < keep.metadata.numel → keep.physicalIndex i < keep.storage.elements
  pointer : 0 < keep.metadata.numel →
    keep.dataPtr = BitVec.ofNat 64 (keep.storage.address keep.storageOffset)
  contiguous : keep.metadata.isContiguous = Bool.true →
    ∀ i, i < keep.metadata.numel → keep.physicalIndex i = keep.storageOffset + i

/-- Signed int32 interpretation retains the raw word's two's-complement sign.
An unresolved negation would negate that word before its logical read. -/
def logicalKeep (read : Nat → BitVec 32) (keep : Tensor) (_laws : Int32Laws keep) (i : Nat) : Int :=
  let raw := read (keep.storage.address (keep.physicalIndex i))
  (if keep.metadata.isNeg then -raw else raw).toInt

def keepView (keep : Tensor) (laws : Int32Laws keep)
    (hc : keep.metadata.isContiguous = Bool.true) : DataView where
  storage := keep.storage
  offset := keep.storageOffset
  elements := keep.metadata.numel
  bounds := by
    intro nonempty
    have last : keep.metadata.numel - 1 < keep.metadata.numel := by omega
    have inside := laws.inBounds _ last
    rw [laws.contiguous hc _ last] at inside
    omega

theorem direct_data_read (read : Nat → Word) (x keep : Tensor) (laws : TensorLaws x)
    (h : TensorContract x.metadata keep.metadata) (i : Nat) (hi : i < x.metadata.numel) :
    read ((dataView x laws h.contiguous.1).address i) = logicalValue read x laws i :=
  direct_read_eq_logical read x laws h.contiguous.1 h.noNeg.1 i hi

theorem keep_address (keep : Tensor) (laws : Int32Laws keep)
    (hc : keep.metadata.isContiguous = Bool.true) (i : Nat) (hi : i < keep.metadata.numel) :
    (keepView keep laws hc).address i = keep.storage.address (keep.physicalIndex i) := by
  rw [laws.contiguous hc i hi]
  simp only [keepView, DataView.address, DataView.base,
    if_neg (by omega : keep.metadata.numel ≠ 0), Storage.address]
  omega

theorem direct_keep_read (read : Nat → BitVec 32) (x keep : Tensor)
    (laws : Int32Laws keep) (h : TensorContract x.metadata keep.metadata)
    (i : Nat) (hi : i < x.metadata.numel) :
    (read ((keepView keep laws h.contiguous.2).address i)).toInt = logicalKeep read keep laws i := by
  have same := same_numel x.metadata keep.metadata h
  rw [keep_address keep laws h.contiguous.2 i (by omega)]
  simp [logicalKeep, h.noNeg.2]

theorem actual_keep_pointer (keep : Tensor) (laws : Int32Laws keep)
    (hc : keep.metadata.isContiguous = Bool.true) (nonempty : 0 < keep.metadata.numel) :
    BitVec.ofNat 64 ((keepView keep laws hc).address 0) = keep.dataPtr := by
  simpa [keepView, DataView.address, DataView.base, Storage.address, Nat.ne_of_gt nonempty]
    using (laws.pointer nonempty).symm

/-- Active mask loads use aligned four-byte addresses inside their backing
allocation, including contiguous views with nonzero storage offsets. -/
theorem keep_active_bytes (x keep : Tensor) (laws : Int32Laws keep)
    (h : TensorContract x.metadata keep.metadata) (i byte : Nat)
    (hi : i < x.metadata.numel) (hb : byte < 4) :
    let address := (keepView keep laws h.contiguous.2).address i
    address % 4 = 0 ∧ keep.storage.base ≤ address + byte ∧
      address + byte < keep.storage.base + 4*keep.storage.elements ∧ address + byte < 2^64 := by
  have same := same_numel x.metadata keep.metadata h
  have hi' : i < keep.metadata.numel := by omega
  have nonempty : 0 < keep.metadata.numel := by omega
  have aligned := (h.alignment (by omega)).2
  change keep.dataPtr.toNat % 4 = 0 at aligned
  rw [← actual_keep_pointer keep laws h.contiguous.2 nonempty] at aligned
  have span := (keepView keep laws h.contiguous.2).spanFits
  have baseBound : (keepView keep laws h.contiguous.2).base < 2^64 := by
    change (keepView keep laws h.contiguous.2).base + 4*keep.metadata.numel ≤ 2^64 at span
    omega
  simp only [DataView.address, Nat.mul_zero, Nat.add_zero,
    BitVec.toNat_ofNat, Nat.mod_eq_of_lt baseBound] at aligned
  have addressAligned : (keepView keep laws h.contiguous.2).address i % 4 = 0 := by
    simp only [DataView.address]
    omega
  have inside := laws.inBounds i hi'
  have storageSpan := keep.storage.spanFits
  dsimp only
  refine ⟨addressAligned, ?_, ?_, ?_⟩ <;>
    rw [keep_address keep laws h.contiguous.2 i hi'] <;>
    simp only [Storage.address] at * <;> omega

#axiomsClean direct_data_read
#axiomsClean direct_keep_read
#axiomsClean actual_keep_pointer
#axiomsClean keep_active_bytes
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DropoutTriton.MemoryContract
