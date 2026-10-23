import bench.repairs.tritonbench_g.matmul_tma_static_checked.KernelOnly.Kernels
import VeriTile.Torch.Tensor

namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticChecked.TensorABI
open VeriTile.Triton FP.Structural FPValue Host VeriTile.Torch
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def width (half : Bool) : Nat := if half then 2 else 4
def torchType (half : Bool) : DType := if half then .other "float16" else .floating .fp32
def stored {α : Type} (d : FloatDType) (x : α) : Cell α := .mk d.toTileDType (ofFloat d x)

def value {α : Type} [Inhabited α] (d : FloatDType) (cell : Cell α) : α :=
  toFloat d (cell.read d.toTileDType)
@[simp] theorem read_stored {α : Type} [Inhabited α] (d : FloatDType) (x : α) :
    value d (stored d x) = x := by cases d <;> rfl
@[simp] theorem half_stored {α : Type} [Inhabited α] (x : α) :
    (stored .fp16 x).read .fp16 = x := rfl

structure Tensor where
  object : Nat
  metadata : TensorMetadata
  (storageId storageBase storageElements storageOffset rowStride columnStride : Nat)
  physicalIndex : Nat → Nat → Nat

/-- Ordinary two-dimensional strided Tensor storage/view laws. -/
structure View (bytes rows columns : Nat) (t : Tensor) : Prop where
  shape : t.metadata.shape = [rows,columns]
  layout : t.metadata.layout = .strided
  resolved : t.metadata.isNeg = Bool.false
  pointer : t.metadata.dataPtr.toNat = t.storageBase + bytes*t.storageOffset
  index : ∀ i j, i < rows → j < columns →
    t.physicalIndex i j = t.storageOffset + i*t.rowStride + j*t.columnStride
  bounds : ∀ i j, i < rows → j < columns → t.physicalIndex i j < t.storageElements

def separated (base₁ size₁ base₂ size₂ : Nat) : Prop :=
  base₁+size₁ ≤ base₂ ∨ base₂+size₂ ≤ base₁

/-- Existing tensor ABI, independent of allocation, successful compilation,
execution, or numerical output. A/B may overlap as read-only views. -/
structure Binding (h : Shape) (f : Flags) (a b c : Tensor) : Prop where
  a_view : View 2 h.rows h.inner a
  b_view : View 2 h.inner h.columns b
  c_view : View (width f.half) h.rows h.columns c
  a_dtype : a.metadata.dtype = torchType Bool.true
  b_dtype : b.metadata.dtype = torchType Bool.true
  c_dtype : c.metadata.dtype = torchType f.half
  device : a.metadata.device.kind = "cuda" ∧ b.metadata.device = a.metadata.device ∧ c.metadata.device = a.metadata.device
  a_strides : a.rowStride = (params h f).aRow ∧ a.columnStride = (params h f).aInner
  b_strides : b.rowStride = (params h f).bInner ∧ b.columnStride = (params h f).bColumn
  c_strides : c.rowStride = h.columns ∧ c.columnStride = 1
  a_window : a.metadata.dataPtr.toNat + 2*(h.rows*h.inner) ≤ 2^64
  b_window : b.metadata.dataPtr.toNat + 2*(h.inner*h.columns) ≤ 2^64
  c_window : c.metadata.dataPtr.toNat + width f.half*(h.rows*h.columns) ≤ 2^64
  a_c : separated a.metadata.dataPtr.toNat (2*(h.rows*h.inner)) c.metadata.dataPtr.toNat (width f.half*(h.rows*h.columns))
  b_c : separated b.metadata.dataPtr.toNat (2*(h.inner*h.columns)) c.metadata.dataPtr.toNat (width f.half*(h.rows*h.columns))

def physicalAddress (bytes : Nat) (t : Tensor) (i j : Nat) :=
  t.storageBase + bytes*t.physicalIndex i j
def logical {α : Type} (read : Nat → α) (bytes : Nat) (t : Tensor) (i j : Nat) : α :=
  read (physicalAddress bytes t i j)

theorem a_address (h : Shape) (f : Flags) (a b c : Tensor) (abi : Binding h f a b c)
    (i : AIndex (params h f)) :
    physicalAddress 2 a i.1.val i.2.1.val = a.metadata.dataPtr.toNat + 2*aOffset (params h f) i := by
  simp only [physicalAddress,abi.a_view.index _ _ i.1.isLt i.2.1.isLt,
    abi.a_strides.1,abi.a_strides.2,abi.a_view.pointer,aOffset]
  omega

theorem b_address (h : Shape) (f : Flags) (a b c : Tensor) (abi : Binding h f a b c)
    (i : BIndex (params h f)) :
    physicalAddress 2 b i.1.val i.2.1.val = b.metadata.dataPtr.toNat + 2*bOffset (params h f) i := by
  simp only [physicalAddress,abi.b_view.index _ _ i.1.isLt i.2.1.isLt,
    abi.b_strides.1,abi.b_strides.2,abi.b_view.pointer,bOffset]
  omega

theorem c_address (h : Shape) (f : Flags) (a b c : Tensor) (abi : Binding h f a b c)
    (i : CIndex (params h f)) :
    physicalAddress (width f.half) c i.1.val i.2.1.val =
      c.metadata.dataPtr.toNat + width f.half*cOffset (params h f) i := by
  simp only [physicalAddress,abi.c_view.index _ _ i.1.isLt i.2.1.isLt,
    abi.c_strides.1,abi.c_strides.2,abi.c_view.pointer,params,Host.c_offset,Nat.mul_one]
  ring

/-- The source regions project the original physical input data. -/
structure Inputs {α : Type} (read : Nat → α) (h : Shape) (a b : Tensor) (s : State α) : Prop where
  left : ∀ off, off < h.rows*h.inner → s.mem "A" off = stored .fp16 (read (a.metadata.dataPtr.toNat+2*off))
  right : ∀ off, off < h.inner*h.columns → s.mem "B" off = stored .fp16 (read (b.metadata.dataPtr.toNat+2*off))

theorem source_inputs {α : Type} [Inhabited α] (read : Nat → α) (h : Shape) (f : Flags)
    (a b c : Tensor) (abi : Binding h f a b c) (s : State α) (input : Inputs read h a b s) :
    (∀ i, aValues (params h f) s i = logical read 2 a i.1.val i.2.1.val) ∧
    (∀ i, bValues (params h f) s i = logical read 2 b i.1.val i.2.1.val) := by
  constructor
  · intro i
    simp only [aValues,regionA,input.left (aOffset (params h f) i) (a_bound h f regions i),half_stored,
      logical,a_address h f a b c abi i]
  · intro i
    simp only [bValues,regionB,input.right (bOffset (params h f) i) (b_bound h f regions i),half_stored,
      logical,b_address h f a b c abi i]

/-- Widened 64-bit physical address used by the block-pointer load/store. -/
def pointer (base bytes off : Nat) : BitVec 64 := BitVec.ofNat 64 (base+bytes*off)
theorem pointer_exact (base bytes count off : Nat) (positive : 0 < bytes)
    (window : base+bytes*count ≤ 2^64) (bound : off < count) :
    (pointer base bytes off).toNat = base+bytes*off := by
  apply Nat.mod_eq_of_lt
  have lt := Nat.mul_lt_mul_of_pos_left bound positive
  omega

theorem source_addresses (h : Shape) (f : Flags) (a b c : Tensor) (abi : Binding h f a b c) :
    (∀ i, (pointer a.metadata.dataPtr.toNat 2 (aOffset (params h f) i)).toNat = physicalAddress 2 a i.1.val i.2.1.val) ∧
    (∀ i, (pointer b.metadata.dataPtr.toNat 2 (bOffset (params h f) i)).toNat = physicalAddress 2 b i.1.val i.2.1.val) ∧
    (∀ i, (pointer c.metadata.dataPtr.toNat (width f.half) (cOffset (params h f) i)).toNat = physicalAddress (width f.half) c i.1.val i.2.1.val) := by
  refine ⟨fun i => ?_,fun i => ?_,fun i => ?_⟩
  · rw [a_address h f a b c abi i]; exact pointer_exact _ _ _ _ (by decide) abi.a_window (a_bound h f regions i)
  · rw [b_address h f a b c abi i]; exact pointer_exact _ _ _ _ (by decide) abi.b_window (b_bound h f regions i)
  · rw [c_address h f a b c abi i]; exact pointer_exact _ _ _ _ (by cases f.half <;> decide) abi.c_window (c_bound h f regions i)

#axiomsClean source_inputs
#axiomsClean source_addresses
end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticChecked.TensorABI
