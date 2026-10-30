import bench.repairs.tritonbench_g.dequantize_rowwise.Proofs.Contract
import VeriTile.Torch.FP32
import VeriTile.Triton.Semantics.State

/-! INT8 storage uses one byte per element. The shared metadata and FP32 scale
binding are reused; the task-local INT8 descriptor supplies the same ordinary
storage/stride/pointer laws. These are API bindings, not kernel-output laws. -/
namespace VeriTile.Bench.Repairs.DequantizeRowwise.InputBinding
open VeriTile.Triton
abbrev Heap := Nat → BitVec 8

namespace Int8
structure Storage where
  object : Nat
  base : Nat
  elements : Nat
  spanFits : base + elements ≤ 2^64
def Storage.address (s : Storage) (i : Nat) : Nat := s.base + i

structure Tensor where
  object : Nat
  metadata : TensorMetadata
  storage : Storage
  storageOffset : Nat
  physicalIndex : Nat → Nat

structure TensorLaws (t : Tensor) : Prop where
  dtype : t.metadata.dtype = .other "int8"
  layout : t.metadata.layout = .strided
  inBounds : ∀ i, i < t.metadata.numel → t.physicalIndex i < t.storage.elements
  pointer : 0 < t.metadata.numel →
    t.metadata.dataPtr = BitVec.ofNat 64 (t.storage.address t.storageOffset)
  contiguous : t.metadata.isContiguous = Bool.true →
    ∀ i, i < t.metadata.numel → t.physicalIndex i = t.storageOffset + i

def logicalValue (heap : Heap) (t : Tensor) (i : Nat) : Int :=
  let raw := heap (t.storage.address (t.physicalIndex i))
  (if t.metadata.isNeg then -raw else raw).toInt

theorem pointer_exact (t : Tensor) (laws : TensorLaws t)
    (dense : t.metadata.isContiguous = Bool.true) (nonempty : 0 < t.metadata.numel) :
    t.metadata.dataPtr.toNat = t.storage.address t.storageOffset := by
  have first := laws.inBounds 0 nonempty
  rw [laws.contiguous dense 0 nonempty] at first
  have span := t.storage.spanFits
  have bound : t.storage.address t.storageOffset < 2^64 := by
    simp only [Storage.address]; omega
  rw [laws.pointer nonempty, BitVec.toNat_ofNat, Nat.mod_eq_of_lt bound]

theorem capacity (t : Tensor) (laws : TensorLaws t)
    (dense : t.metadata.isContiguous = Bool.true) (nonempty : 0 < t.metadata.numel) :
    t.metadata.dataPtr.toNat + t.metadata.numel ≤ 2^64 := by
  have last : t.metadata.numel - 1 < t.metadata.numel := by omega
  have bound := laws.inBounds _ last
  rw [laws.contiguous dense _ last] at bound
  rw [pointer_exact t laws dense nonempty]
  have span := t.storage.spanFits
  simp only [Storage.address]; omega

theorem direct_read (heap : Heap) (t : Tensor) (laws : TensorLaws t)
    (dense : t.metadata.isContiguous = Bool.true) (resolved : t.metadata.isNeg = Bool.false)
    (i : Nat) (hi : i < t.metadata.numel) :
    (heap (t.metadata.dataPtr.toNat + i)).toInt = logicalValue heap t i := by
  rw [pointer_exact t laws dense (by omega)]
  simp [logicalValue, resolved, laws.contiguous dense i hi, Storage.address, Nat.add_assoc]

theorem signed_range (word : BitVec 8) : -128 ≤ word.toInt ∧ word.toInt ≤ 127 :=
  ⟨by simpa using BitVec.le_toInt word, by simpa using (BitVec.toInt_le (x := word))⟩
end Int8

structure Request where
  x : Int8.Tensor
  scale : VeriTile.Torch.FP32.Tensor
  xLaws : Int8.TensorLaws x
  scaleLaws : VeriTile.Torch.FP32.TensorLaws scale
  contract : InputContract x.metadata scale.metadata

def scaleView (c : Request) :=
  VeriTile.Torch.FP32.dataView c.scale c.scaleLaws c.contract.contiguous.2

theorem scale_pointer (c : Request) (nonempty : 0 < c.scale.metadata.numel) :
    (scaleView c).base = c.scale.metadata.dataPtr.toNat := by
  have hp := VeriTile.Torch.FP32.actual_pointer c.scale c.scaleLaws c.contract.contiguous.2 nonempty
  have span := (scaleView c).spanFits
  have hb : (scaleView c).base < 2^64 := by
    change (scaleView c).base + 4 * c.scale.metadata.numel ≤ 2^64 at span
    omega
  have hn := congrArg BitVec.toNat hp
  change (scaleView c).address 0 % 2^64 = c.scale.metadata.dataPtr.toNat at hn
  simpa only [VeriTile.Torch.FP32.DataView.address, Nat.mul_zero, Nat.add_zero,
    Nat.mod_eq_of_lt hb] using hn

theorem scale_aligned (c : Request) (nonempty : 0 < c.x.metadata.numel) :
    (scaleView c).base % 4 = 0 := by
  have productPositive : 0 < rows c.x.metadata * columns c.x.metadata := by
    simpa [numel_eq _ _ c.contract] using nonempty
  have scalePositive : 0 < c.scale.metadata.numel := by
    have rowPositive : 0 < rows c.x.metadata := Nat.pos_of_mul_pos_right productPositive
    exact lt_of_lt_of_le rowPositive (c.contract.scales nonempty)
  rw [scale_pointer c scalePositive]
  exact c.contract.alignment nonempty

/-- Little-endian FP32 components read from the same heap as INT8 weights.
Read-only views may overlap, including across different input dtypes. -/
def readScale (heap : Heap) (address : Nat) : VeriTile.Torch.FP32.Word :=
  FP.Value.ofBits .fp32 ((heap address).toNat + 256 * (heap (address + 1)).toNat +
    65536 * (heap (address + 2)).toNat + 16777216 * (heap (address + 3)).toNat)

noncomputable def initial (heap : Heap) (interpret : VeriTile.Torch.FP32.Word → ℝ)
    (c : Request) (scratch : BlockState) : BlockState :=
  { scratch with mem := fun r i =>
      if r = xRegion then MemCell.of .int (heap (c.x.metadata.dataPtr.toNat + i)).toInt
      else if r = scaleRegion then MemCell.real (interpret (readScale heap ((scaleView c).address i)))
      else scratch.mem r i }

theorem input_read (heap : Heap) (interpret : VeriTile.Torch.FP32.Word → ℝ)
    (c : Request) (scratch : BlockState) (i : Nat) (hi : i < c.x.metadata.numel) :
    (initial heap interpret c scratch).readMemValue .int xRegion i = Int8.logicalValue heap c.x i := by
  simpa only [initial, BlockState.readMemValue, BlockState.readMemTyped,
    ↓reduceIte, MemCell.readAs_of_same] using
    Int8.direct_read heap c.x c.xLaws c.contract.contiguous.1 c.contract.noNeg.1 i hi

theorem scale_read (heap : Heap) (interpret : VeriTile.Torch.FP32.Word → ℝ)
    (c : Request) (scratch : BlockState) (i : Nat) (hi : i < c.scale.metadata.numel) :
    (initial heap interpret c scratch).readMem scaleRegion i =
      interpret (VeriTile.Torch.FP32.logicalValue (readScale heap) c.scale c.scaleLaws i) := by
  have h := VeriTile.Torch.FP32.direct_read_eq_logical (readScale heap) c.scale c.scaleLaws
    c.contract.contiguous.2 c.contract.noNeg.2 i hi
  simpa [initial, xRegion, scaleRegion, BlockState.readMem, MemCell.readAs, MemCell.real,
    MemCell.of, scaleView] using congrArg interpret h

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DequantizeRowwise.InputBinding
