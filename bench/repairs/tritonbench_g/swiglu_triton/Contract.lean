import VeriTile.Torch.Tensor
import bench.repairs.tritonbench_g.swiglu_triton.Proofs.DirectPhysical
import Mathlib.Data.Nat.Log

namespace VeriTile.Bench.Repairs.SwigluTriton.DirectHost
open Memory DirectPhysical
open VeriTile.Triton

/-- Ordinary Tensor API binding. The logical element locations are independent
of execution. Contiguity binds them to the queried data pointer, including the
actual storage offset; lazy negation remains a separate metadata flag. -/
structure Tensor where
  object : Nat
  metadata : VeriTile.Torch.TensorMetadata
  key : Nat → Key
  contiguousKeys : metadata.isStrided = Bool.true → metadata.isFP32 = Bool.true →
    metadata.isContiguous = Bool.true → ∀ i, i < metadata.numel →
    key i = ⟨metadata.device.index,metadata.dataPtr.toNat+4*i⟩
  contiguousSpan : metadata.isStrided = Bool.true → metadata.isFP32 = Bool.true →
    metadata.isContiguous = Bool.true →
    metadata.dataPtr.toNat+4*metadata.numel ≤ 2^64

def Tensor.read {M α β : Type} (t : Tensor) (ops : Ops M α β) (s : M) (i : Nat) : α :=
  ops.read s (t.key i)

def payload (t : Tensor) : Bool :=
  t.metadata.isStrided && t.metadata.isFP32 && t.metadata.isCuda &&
    t.metadata.isContiguous && !t.metadata.isNeg

def spanSeparate (a b : Tensor) : Prop :=
  a.metadata.dataPtr.toNat+4*a.metadata.numel ≤ b.metadata.dataPtr.toNat ∨
  b.metadata.dataPtr.toNat+4*b.metadata.numel ≤ a.metadata.dataPtr.toNat

instance (a b : Tensor) : Decidable (spanSeparate a b) := inferInstanceAs (Decidable (_ ∨ _))

/-- Ordinary source view/empty_like receipt. No output value or successful
kernel result can be supplied. For forward, third is the fresh empty_like
allocation; for backward it is the caller's upstream tensor. -/
structure Prepared (backward : Bool) where
  tensor : Role → Tensor
  originalShape : List Nat
  rows : Nat
  cols : Nat
  shapeView : originalShape.prod = rows*cols
  lastDimension : originalShape.getLastD 0 = cols
  shaped : (if backward then tensor .third else tensor .a).metadata.shape = originalShape
  allocatedCount : backward = Bool.false → (tensor .third).metadata.numel = rows*cols
  allocatedPayload : backward = Bool.false → payload (tensor .third) = Bool.true
  allocatedDevice : backward = Bool.false → (tensor .third).metadata.device = (tensor .a).metadata.device
  allocatedAlignment : backward = Bool.false → 0 < rows*cols → (tensor .third).metadata.dataPtr.toNat % 4 = 0
  allocatedSeparate : backward = Bool.false → 0 < rows*cols →
    spanSeparate (tensor .a) (tensor .third) ∧ spanSeparate (tensor .b) (tensor .third)
  returnedObject : Role → Nat

instance : Fintype Role where
  elems := {.a,.b,.third}
  complete r := by cases r <;> simp

/-- These are the actual checked launch inputs; forward's allocator supplies
third's facts by the ordinary empty_like contract. -/
structure Admitted (p : Prepared backward) : Prop where
  payloads : ∀ r, payload (p.tensor r) = Bool.true
  counts : ∀ r, (p.tensor r).metadata.numel = p.rows*p.cols
  forwardShape : backward = Bool.false → (p.tensor .b).metadata.shape = p.originalShape
  sameDevice : ∀ r, (p.tensor r).metadata.device = (p.tensor .a).metadata.device
  positive : 0 < p.cols
  rows32 : p.rows < 2^31
  padded64 : p.rows*p.cols+65536 < 2^63
  alignment : 0 < p.rows*p.cols → ∀ r, (p.tensor r).metadata.dataPtr.toNat % 4 = 0
  separate : 0 < p.rows*p.cols → ∀ r s, r ≠ s → spanSeparate (p.tensor r) (p.tensor s)

def check (p : Prepared backward) : Bool :=
  decide (∀ r, payload (p.tensor r) = Bool.true) &&
  decide (∀ r, (p.tensor r).metadata.numel = p.rows*p.cols) &&
  decide (backward = Bool.false → (p.tensor .b).metadata.shape = p.originalShape) &&
  decide (∀ r, (p.tensor r).metadata.device = (p.tensor .a).metadata.device) &&
  decide (0 < p.cols) && decide (p.rows < 2^31) && decide (p.rows*p.cols+65536 < 2^63) &&
  decide (0 < p.rows*p.cols → ∀ r, (p.tensor r).metadata.dataPtr.toNat % 4 = 0) &&
  decide (0 < p.rows*p.cols → ∀ r s, r ≠ s → spanSeparate (p.tensor r) (p.tensor s))

def blockSize (cols : Nat) : Nat := 2^Nat.clog 2 cols

end VeriTile.Bench.Repairs.SwigluTriton.DirectHost
