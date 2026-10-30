import VeriTile.Torch.Tensor
import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.HostTypes

namespace VeriTile.Bench.Repairs.TritonLinearActivation.Checked
open VeriTile.Torch
noncomputable section
open Classical
set_option autoImplicit false

/-- Production source pin; the historical candidate pin remains in Kernels.lean. -/
def checkedSHA256 : String := "271cfcf8614afc2747837dbf8902f3addc9a3a605a6c1a85063ae2b6c346f8a9"

/-- Direct observations of a caller argument. A non-Tensor is classified
before any of the placeholder Tensor fields are queried. -/
structure Observed where
  isTensor : Bool
  metadata : TensorMetadata
  strides : List Nat
  offset : Nat
  conjugate : Bool
  base : Nat
  bytes : Nat
  object : Nat
  deriving DecidableEq

def Used (r : Request) (i : Fin 4) : Prop :=
  i.val < 2 ∨ (i = 2 ∧ r.hasBias = Bool.true) ∨ (i = 3 ∧ r.hasSaved = Bool.true)

def Canonical (owner : Device) (t : Observed) : Prop :=
  t.isTensor = Bool.true ∧ t.metadata.layout = .strided ∧ t.metadata.device.kind = "cuda" ∧
  t.metadata.device = owner ∧ t.metadata.dtype = .floating .fp32 ∧
  t.metadata.isContiguous = Bool.true ∧ t.offset = 0 ∧ t.metadata.isNeg = Bool.false ∧
  t.conjugate = Bool.false ∧
  (t.metadata.numel ≠ 0 → t.metadata.dataPtr.toNat % 4 = 0) ∧
  t.strides = canonicalStrides t.metadata.shape

def Separate (a b : Observed) : Prop := a.base+a.bytes ≤ b.base ∨ b.base+b.bytes ≤ a.base

/-- Shape extraction and identity observations, with no source value or
execution premise. Optional argument flags and ambient AMP mode belong to r. -/
structure Binding (r : Request) (ts : Fin 4 → Observed) : Prop where
  storage : ∀ i, r.storage i = (ts i).base
  object : ∀ i, r.object i = (ts i).object
  device : r.device = (ts 0).metadata.device.index
  nonemptyBytes : ∀ i, Used r i → (ts i).metadata.numel ≠ 0 → 0 < (ts i).bytes

structure Domain (r : Request) (ts : Fin 4 → Observed) (activationIsValid : Bool) : Prop where
  activation : activationIsValid = Bool.true
  canonical : ∀ i, Used r i → Canonical (ts 0).metadata.device (ts i)
  shape : ∀ i, Used r i → (ts i).metadata.shape = r.argumentShape i
  separate : ∀ i j, Used r i → Used r j → i < j →
    (ts i).metadata.numel ≠ 0 → (ts j).metadata.numel ≠ 0 → Separate (ts i) (ts j)
  bounds : SourceBounds r.shape

/-- Accepted-domain checker after ordinary rank/dimension extraction. The
metadata predicates are grouped; Python exception-label order is not modeled.
Every predicate is checked before any
allocation, cast, device entry, launch, or destination store. -/
def check (r : Request) (ts : Fin 4 → Observed) (activationIsValid : Bool) : Except String Unit :=
  if activationIsValid ≠ Bool.true then .error "activation type"
  else if ¬(∀ i, Used r i → Canonical (ts 0).metadata.device (ts i)) then .error "canonical Tensor"
  else if ¬(∀ i, Used r i → (ts i).metadata.shape = r.argumentShape i) then .error "shape"
  else if ¬(∀ i j, Used r i → Used r j → i < j →
      (ts i).metadata.numel ≠ 0 → (ts j).metadata.numel ≠ 0 → Separate (ts i) (ts j)) then .error "allocation alias"
  else if ¬ SourceBounds r.shape then .error "formed source indices" else .ok ()

theorem python_checks_iff (r : Request) (ts : Fin 4 → Observed) (activationIsValid : Bool) :
    check r ts activationIsValid = .ok () ↔ Domain r ts activationIsValid := by
  constructor
  · intro checked
    unfold check at checked
    split_ifs at checked with activation canonical shape separate bounds <;> try cases checked
    exact ⟨by simpa using activation,canonical,shape,separate,bounds⟩
  · intro h
    unfold check
    rw [if_neg (by simpa using h.activation),if_neg (not_not.mpr h.canonical),
      if_neg (not_not.mpr h.shape),if_neg (not_not.mpr h.separate),if_neg (not_not.mpr h.bounds)]

theorem used_count (r : Request) (i : Fin 4) : 0 < r.count i → Used r i := by
  fin_cases i <;> simp [Request.count,Used] <;> split <;> simp_all

theorem bound_numel (r : Request) (ts : Fin 4 → Observed) (h : Binding r ts)
    (i : Fin 4) (used : Used r i) (shape : (ts i).metadata.shape = r.argumentShape i) :
    (ts i).metadata.numel = r.count i := by
  rw [TensorMetadata.numel,shape]
  fin_cases i <;> cases bias : r.hasBias <;> cases saved : r.hasSaved <;>
    cases batch : r.batch <;> simp_all [Used,Request.argumentShape,Request.count,Batch.shape,Batch.rows,Nat.mul_assoc]

theorem domain_valid (r : Request) (ts : Fin 4 → Observed) (h : Binding r ts) {activationIsValid : Bool}
    (admit : Domain r ts activationIsValid) : r.Valid := by
  refine ⟨admit.bounds,?_⟩
  intro i j different hi hj same
  have ui := used_count r i hi
  have uj := used_count r j hj
  have ni : (ts i).metadata.numel ≠ 0 := by rw [bound_numel r ts h i ui (admit.shape i ui)];omega
  have nj : (ts j).metadata.numel ≠ 0 := by rw [bound_numel r ts h j uj (admit.shape j uj)];omega
  have bi := h.nonemptyBytes i ui ni
  have bj := h.nonemptyBytes j uj nj
  rw [h.storage i,h.storage j] at same
  rcases lt_or_gt_of_ne different with order | order
  · rcases admit.separate i j ui uj order ni nj with left | right <;> omega
  · rcases admit.separate j i uj ui order nj ni with left | right <;> omega

#axiomsClean python_checks_iff
#axiomsClean domain_valid
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonLinearActivation.Checked
