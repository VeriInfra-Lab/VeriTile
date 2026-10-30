import bench.tritonbench_g.masked_add_cuda.Proofs.Contract
import VeriTile.Torch.FP32

/-! Input-only storage bindings. A single little-endian byte reader represents
all three tensors, so readonly FP32/int32/int64 views may overlap. The shared
FP32 Tensor laws connect metadata with actual pointers and logical elements.
Integer masks retain their own width and signed interpretation. -/
namespace VeriTile.Bench.Repairs.MaskedAddCuda.MemoryContract
open VeriTile Triton
open VeriTile.Torch.FP32

abbrev Bytes := Nat → BitVec 8

def readWord (read : Bytes) (bytes address : Nat) : BitVec (8*bytes) :=
  BitVec.ofNat (8*bytes) ((List.range bytes).foldl
    (fun value byte => value + (read (address+byte)).toNat * 256^byte) 0)

def readFP32 (read : Bytes) (address : Nat) : Word := ⟨readWord read 4 address⟩

/-- Integer tensor storage geometry uses its selected four/eight-byte width;
it is not assigned the four-byte FP32 storage interpretation. -/
structure MaskTensor where
  metadata : TensorMetadata
  storageBase : Nat
  storageElements : Nat
  storageOffset : Nat
  physicalIndex : Nat → Nat

def MaskTensor.width (m : MaskTensor) : Nat := maskBytes (selectedMask m.metadata)
def MaskTensor.address (m : MaskTensor) (i : Nat) : Nat := m.storageBase + m.width*i

structure MaskLaws (m : MaskTensor) : Prop where
  dtype : m.metadata.dtype = maskDType (selectedMask m.metadata)
  layout : m.metadata.layout = .strided
  span : m.storageBase + m.width*m.storageElements ≤ 2^64
  inBounds : ∀ i, i < m.metadata.numel → m.physicalIndex i < m.storageElements
  pointer : 0 < m.metadata.numel → m.metadata.dataPtr = BitVec.ofNat 64 (m.address m.storageOffset)
  contiguous : m.metadata.isContiguous = Bool.true →
    ∀ i, i < m.metadata.numel → m.physicalIndex i = m.storageOffset+i

def logicalMask (read : Bytes) (m : MaskTensor) (_laws : MaskLaws m) (i : Nat) : Int :=
  let raw := readWord read m.width (m.address (m.physicalIndex i))
  (if m.metadata.isNeg then -raw else raw).toInt

theorem fp32_address (t : Tensor) (laws : TensorLaws t)
    (hc : t.metadata.isContiguous = Bool.true) (i : Nat) (hi : i < t.metadata.numel) :
    t.metadata.dataPtr.toNat + 4*i = (dataView t laws hc).address i := by
  have nonempty : 0 < t.metadata.numel := by omega
  have span := (dataView t laws hc).spanFits
  have baseBound : (dataView t laws hc).base < 2^64 := by
    change (dataView t laws hc).base + 4*t.metadata.numel ≤ 2^64 at span
    omega
  have pointer := actual_pointer t laws hc nonempty
  change BitVec.ofNat 64 ((dataView t laws hc).address 0) = t.metadata.dataPtr at pointer
  rw [←pointer]
  simp only [DataView.address, Nat.mul_zero, Nat.add_zero, BitVec.toNat_ofNat,
    Nat.mod_eq_of_lt baseBound]

theorem mask_address (m : MaskTensor) (laws : MaskLaws m)
    (hc : m.metadata.isContiguous = Bool.true) (i : Nat) (hi : i < m.metadata.numel) :
    m.metadata.dataPtr.toNat + m.width*i = m.address (m.physicalIndex i) := by
  have positive : 0 < m.width := by
    unfold MaskTensor.width
    cases selectedMask m.metadata <;> decide
  have inside := laws.inBounds i hi
  rw [laws.contiguous hc i hi] at inside
  have scaled := Nat.mul_lt_mul_of_pos_left inside positive
  have smaller := Nat.mul_le_mul_left m.width (Nat.le_add_right m.storageOffset i)
  have span := laws.span
  have bound : m.address m.storageOffset < 2^64 := by
    unfold MaskTensor.address
    omega
  rw [laws.pointer (by omega), BitVec.toNat_ofNat, Nat.mod_eq_of_lt bound,
    laws.contiguous hc i hi]
  simp [MaskTensor.address, Nat.mul_add, Nat.add_assoc]

theorem direct_fp32_read (read : Bytes) (t : Tensor) (laws : TensorLaws t)
    (hc : t.metadata.isContiguous = Bool.true) (hn : t.metadata.isNeg = Bool.false)
    (i : Nat) (hi : i < t.metadata.numel) :
    readFP32 read (t.metadata.dataPtr.toNat+4*i) = logicalValue (readFP32 read) t laws i := by
  rw [fp32_address t laws hc i hi]
  exact direct_read_eq_logical (readFP32 read) t laws hc hn i hi

theorem direct_mask_read (read : Bytes) (m : MaskTensor) (laws : MaskLaws m)
    (hc : m.metadata.isContiguous = Bool.true) (hn : m.metadata.isNeg = Bool.false)
    (i : Nat) (hi : i < m.metadata.numel) :
    (readWord read m.width (m.metadata.dataPtr.toNat+m.width*i)).toInt = logicalMask read m laws i := by
  rw [mask_address m laws hc i hi]
  simp [logicalMask, hn]

/-- Contract-to-memory connection using one common backing reader. No separation
between data and mask is assumed, including mixed-width overlapping views. -/
theorem checked_tensor_reads (read : Bytes) (g p : Tensor) (m : MaskTensor)
    (lg : TensorLaws g) (lp : TensorLaws p) (lm : MaskLaws m)
    (a : AlphaArgument) (h : InputContract g.metadata p.metadata m.metadata a)
    (i : Nat) (hi : i < g.metadata.numel) :
    readFP32 read (g.metadata.dataPtr.toNat+4*i) = logicalValue (readFP32 read) g lg i ∧
    readFP32 read (p.metadata.dataPtr.toNat+4*i) = logicalValue (readFP32 read) p lp i ∧
    (readWord read m.width (m.metadata.dataPtr.toNat+m.width*i)).toInt = logicalMask read m lm i := by
  rcases h with ⟨_, _, _, _, shapes, contiguous, noNeg, _⟩
  have np : g.metadata.numel = p.metadata.numel := congrArg List.prod shapes.1
  have nm : g.metadata.numel = m.metadata.numel := congrArg List.prod shapes.2
  exact ⟨direct_fp32_read read g lg contiguous.1 noNeg.1 i hi,
    direct_fp32_read read p lp contiguous.2.1 noNeg.2.1 i (by omega),
    direct_mask_read read m lm contiguous.2.2 noNeg.2.2 i (by omega)⟩

#axiomsClean fp32_address
#axiomsClean mask_address
#axiomsClean direct_fp32_read
#axiomsClean direct_mask_read
#axiomsClean checked_tensor_reads
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.MaskedAddCuda.MemoryContract
