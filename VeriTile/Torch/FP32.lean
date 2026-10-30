import VeriTile.Torch.Tensor
import VeriTile.Meta.StatementAudit
import Mathlib.Data.Nat.Basic
import Lean.Elab.Tactic.Omega

/-! FP32 storage bindings and Tensor API semantics. A binding certifies dtype,
layout and the logical/physical address relation. Data views retain their
backing storage; deriving a view neither allocates nor changes storage identity.
No law assumes a kernel execution, arithmetic result or grid. -/
namespace VeriTile.Torch.FP32
open VeriTile.Triton

abbrev Word := FP.Value .fp32

/-- One underlying FP32 storage allocation, with its own stable identity. -/
structure Storage where
  object : Nat
  base : Nat
  elements : Nat
  spanFits : base + 4 * elements ≤ 2^64

def Storage.address (s : Storage) (i : Nat) : Nat := s.base + 4*i

/-- A contiguous window into existing storage. Empty windows need no bound on
an unused offset; they still retain the identity of their backing storage. -/
structure DataView where
  storage : Storage
  offset : Nat
  elements : Nat
  bounds : 0 < elements → offset + elements ≤ storage.elements

namespace DataView

def base (v : DataView) : Nat :=
  if v.elements = 0 then 0 else v.storage.address v.offset

def address (v : DataView) (i : Nat) : Nat := v.base + 4*i

theorem spanFits (v : DataView) : v.base + 4*v.elements ≤ 2^64 := by
  unfold base
  split
  · rename_i empty
    simp [empty]
  · rename_i nonempty
    have cap := v.bounds (Nat.pos_of_ne_zero nonempty)
    have span := v.storage.spanFits
    simp only [Storage.address]
    omega

end DataView

/-- Tensor object identity is distinct from backing storage identity. -/
structure Tensor where
  object : Nat
  metadata : TensorMetadata
  storage : Storage
  storageOffset : Nat
  physicalIndex : Nat → Nat

/-- Admission and the native binding use the same observed pointer. -/
def Tensor.dataPtr (t : Tensor) : BitVec 64 := t.metadata.dataPtr

/-- A valid binding of strided FP32 Tensor metadata to storage. Dtype/layout
compatibility is explicit, so an arbitrary descriptor cannot be interpreted as
FP32 merely by belonging to this namespace. Ordinary API laws describe element
bounds, actual pointers and contiguity; empty tensors have no unused bounds.
API references: [data_ptr](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.data_ptr.html)
observes the first element's address;
[storage_offset](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.storage_offset.html)
counts storage elements; [is_contiguous](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.is_contiguous.html)
uses the requested memory format, here the default contiguous format.
These are explicit API-binding contracts, not a proof of Torch's implementation
or a guarantee that every backing storage has aligned addresses. -/
structure TensorLaws (t : Tensor) : Prop where
  dtype : t.metadata.dtype = .floating .fp32
  layout : t.metadata.layout = .strided
  inBounds : ∀ i, i < t.metadata.numel → t.physicalIndex i < t.storage.elements
  pointer : 0 < t.metadata.numel →
    t.dataPtr = BitVec.ofNat 64 (t.storage.address t.storageOffset)
  contiguous : t.metadata.isContiguous = Bool.true →
    ∀ i, i < t.metadata.numel → t.physicalIndex i = t.storageOffset + i

/-- Inconsistent dtype descriptors have no valid FP32 storage binding. -/
theorem no_binding_of_dtype_ne (t : Tensor) (h : t.metadata.dtype ≠ .floating .fp32) :
    ¬ TensorLaws t := fun laws => h laws.dtype

theorem no_binding_of_layout_ne (t : Tensor) (h : t.metadata.layout ≠ .strided) :
    ¬ TensorLaws t := fun laws => h laws.layout

/-- The span bound is derived from element bounds and checked contiguity. -/
theorem contiguous_capacity (t : Tensor) (laws : TensorLaws t)
    (hc : t.metadata.isContiguous = Bool.true) (hne : 0 < t.metadata.numel) :
    t.storageOffset + t.metadata.numel ≤ t.storage.elements := by
  have hi : t.metadata.numel - 1 < t.metadata.numel := by omega
  have bound := laws.inBounds (t.metadata.numel - 1) hi
  rw [laws.contiguous hc _ hi] at bound
  omega

def dataView (t : Tensor) (laws : TensorLaws t)
    (hc : t.metadata.isContiguous = Bool.true) : DataView where
  storage := t.storage
  offset := t.storageOffset
  elements := t.metadata.numel
  bounds := contiguous_capacity t laws hc

@[simp] theorem dataView_storage (t : Tensor) (laws : TensorLaws t)
    (hc : t.metadata.isContiguous = Bool.true) :
    (dataView t laws hc).storage = t.storage := rfl

/-- Distinct Tensor objects sharing storage retain that storage in their views. -/
theorem dataView_same_storage (x y : Tensor) (lx : TensorLaws x) (ly : TensorLaws y)
    (cx : x.metadata.isContiguous = Bool.true) (cy : y.metadata.isContiguous = Bool.true)
    (h : x.storage = y.storage) : (dataView x lx cx).storage = (dataView y ly cy).storage := h

/-- Logical FP32 interpretation requires a valid storage binding. The negative
view flag changes logical values without requiring materialized negation; see
[torch.resolve_neg](https://docs.pytorch.org/docs/stable/generated/torch.resolve_neg.html).
The direct-read theorem below therefore requires an absent negative flag. -/
def logicalValue (read : Nat → Word) (t : Tensor) (_laws : TensorLaws t) (i : Nat) : Word :=
  let raw := read (t.storage.address (t.physicalIndex i))
  if t.metadata.isNeg then raw.neg else raw

theorem direct_read_eq_logical (read : Nat → Word) (t : Tensor)
    (laws : TensorLaws t) (hc : t.metadata.isContiguous = Bool.true)
    (notNeg : t.metadata.isNeg = Bool.false) (i : Nat) (hi : i < t.metadata.numel) :
    read ((dataView t laws hc).address i) = logicalValue read t laws i := by
  simp only [logicalValue, notNeg, Bool.false_eq_true, ↓reduceIte,
    laws.contiguous hc i hi]
  apply congrArg read
  simp only [DataView.address, DataView.base, dataView,
    if_neg (by omega : t.metadata.numel ≠ 0), Storage.address]
  omega

theorem actual_pointer (t : Tensor) (laws : TensorLaws t)
    (hc : t.metadata.isContiguous = Bool.true) (hne : 0 < t.metadata.numel) :
    BitVec.ofNat 64 ((dataView t laws hc).address 0) = t.dataPtr := by
  simpa [dataView, DataView.address, DataView.base, Storage.address, Nat.ne_of_gt hne]
    using (laws.pointer hne).symm

/-- Access alignment comes from the checked data pointer, not an unconditional
property of Tensor storage. Empty windows access no bytes. -/
theorem dataView_aligned (t : Tensor) (laws : TensorLaws t)
    (hc : t.metadata.isContiguous = Bool.true)
    (checked : 0 < t.metadata.numel → t.dataPtr.toNat % 4 = 0) :
    (dataView t laws hc).base % 4 = 0 := by
  by_cases empty : t.metadata.numel = 0
  · simp [dataView, DataView.base, empty]
  have nonempty : 0 < t.metadata.numel := Nat.pos_of_ne_zero empty
  have aligned := checked nonempty
  rw [←actual_pointer t laws hc nonempty] at aligned
  have span := (dataView t laws hc).spanFits
  have bound : (dataView t laws hc).base < 2^64 := by
    change (dataView t laws hc).base + 4*t.metadata.numel ≤ 2^64 at span
    omega
  simpa only [DataView.address, Nat.mul_zero, Nat.add_zero,
    BitVec.toNat_ofNat, Nat.mod_eq_of_lt bound] using aligned

#axiomsClean no_binding_of_dtype_ne
#axiomsClean no_binding_of_layout_ne
#axiomsClean contiguous_capacity
#axiomsClean dataView_same_storage
#axiomsClean direct_read_eq_logical
#axiomsClean actual_pointer
#axiomsClean dataView_aligned
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Torch.FP32
