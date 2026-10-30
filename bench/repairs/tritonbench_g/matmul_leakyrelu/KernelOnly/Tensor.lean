import VeriTile.Torch.FP32
import bench.repairs.tritonbench_g.matmul_leakyrelu.Proofs.Contract

/-! Ordinary bindings of actual input Tensor objects and the FP16 output.
The strided indexing law preserves transposes, slices, broadcasts, and input
aliasing. It contains no execution or arithmetic-result assumption. -/
namespace VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatmulLeakyrelu
noncomputable section
set_option maxHeartbeats 2000000

structure Matrix where
  tensor : VeriTile.Torch.FP32.Tensor
  (rows columns strideRow strideColumn : Nat)
  /-- Untyped backing size includes any trailing bytes after the final FP32 word. -/
  storageBytes : Nat

structure MatrixLaws (x : Matrix) : Prop where
  tensor : VeriTile.Torch.FP32.TensorLaws x.tensor
  shape : x.tensor.metadata.shape = [x.rows, x.columns]
  indices : ∀ i < x.rows, ∀ j < x.columns,
    x.tensor.physicalIndex (i*x.columns+j) =
      x.tensor.storageOffset+i*x.strideRow+j*x.strideColumn
  emptyPointer : x.tensor.metadata.numel = 0 → x.tensor.dataPtr.toNat = 0
  backingElements : x.storageBytes / 4 = x.tensor.storage.elements
  backingSpan : x.tensor.storage.base + x.storageBytes ≤ 2^64

def metadata (a b : Matrix) : Metadata :=
  ⟨a.tensor.metadata, b.tensor.metadata, a.strideRow, a.strideColumn,
    b.strideRow, b.strideColumn⟩

def request (a b : Matrix) (activation : String) : Whole.Request where
  A := "a"; B := "b"; C := "c"
  rows := a.rows; inner := a.columns; innerB := b.rows; columns := b.columns
  strideAM := a.strideRow; strideAK := a.strideColumn
  strideBK := b.strideRow; strideBN := b.strideColumn
  aContiguous := a.tensor.metadata.isContiguous
  activation := activation

theorem request_eq (a b : Matrix) (al : MatrixLaws a) (bl : MatrixLaws b)
    (activation : String) :
    MatmulLeakyrelu.request (metadata a b) activation = request a b activation := by
  simp [MatmulLeakyrelu.request, metadata, request, al.shape, bl.shape]

theorem numel (x : Matrix) (xl : MatrixLaws x) :
    x.tensor.metadata.numel = x.rows*x.columns := by
  simp [VeriTile.Torch.TensorMetadata.numel, xl.shape]

theorem flat_bound (rows cols i j : Nat) (hi : i < rows) (hj : j < cols) :
    i*cols+j < rows*cols := by nlinarith

def inputBase (x : Matrix) : Nat := x.tensor.dataPtr.toNat
def inputExtent (x : Matrix) : Nat :=
  if x.tensor.metadata.numel = 0 then 0 else x.tensor.storage.elements-x.tensor.storageOffset
def logical {α : Type} (read : Nat → α) (x : Matrix) (i j : Nat) : α :=
  read (x.tensor.storage.address (x.tensor.physicalIndex (i*x.columns+j)))

theorem input_address (x : Matrix) (xl : MatrixLaws x) (i j : Nat)
    (hi : i < x.rows) (hj : j < x.columns) :
    x.tensor.storage.address (x.tensor.physicalIndex (i*x.columns+j)) =
      inputBase x + 4*(i*x.strideRow+j*x.strideColumn) := by
  have idx := flat_bound x.rows x.columns i j hi hj
  have hnum : 0 < x.tensor.metadata.numel := by rw [numel x xl]; omega
  have bound := xl.tensor.inBounds _ (by simpa [numel x xl] using idx)
  rw [xl.indices i hi j hj] at bound
  have span := x.tensor.storage.spanFits
  have hb : x.tensor.storage.address x.tensor.storageOffset < 2^64 := by
    unfold VeriTile.Torch.FP32.Storage.address
    omega
  have hp := xl.tensor.pointer hnum
  simp only [inputBase, hp, BitVec.toNat_ofNat, Nat.mod_eq_of_lt hb,
    xl.indices i hi j hj, VeriTile.Torch.FP32.Storage.address]
  omega

theorem input_capacity (x : Matrix) (xl : MatrixLaws x) (i j : Nat)
    (hi : i < x.rows) (hj : j < x.columns) :
    i*x.strideRow+j*x.strideColumn < inputExtent x := by
  have idx := flat_bound x.rows x.columns i j hi hj
  have hn : x.tensor.metadata.numel ≠ 0 := by rw [numel x xl]; omega
  have hb := xl.tensor.inBounds _ (by simpa [numel x xl] using idx)
  rw [xl.indices i hi j hj] at hb
  simp only [inputExtent, hn, if_false]
  omega

theorem input_span (x : Matrix) (xl : MatrixLaws x) :
    inputBase x + 4*inputExtent x ≤ 2^64 := by
  by_cases hz : x.tensor.metadata.numel = 0
  · simp [inputBase, inputExtent, hz, xl.emptyPointer hz]
  have hn : 0 < x.rows*x.columns := by rw [←numel x xl]; omega
  have hr : 0 < x.rows := by nlinarith
  have hc : 0 < x.columns := by nlinarith
  have cap := input_capacity x xl 0 0 hr hc
  have addr := input_address x xl 0 0 hr hc
  have span := x.tensor.storage.spanFits
  rw [xl.indices 0 hr 0 hc] at addr
  simp only [Nat.zero_mul, Nat.add_zero, Nat.zero_add] at cap addr
  simp only [VeriTile.Torch.FP32.Storage.address] at addr
  simp only [inputExtent, hz, if_false] at cap ⊢
  omega

theorem logical_fp32 (read : Nat → VeriTile.Torch.FP32.Word)
    (x : Matrix) (xl : MatrixLaws x) (notNeg : x.tensor.metadata.isNeg = Bool.false)
    (i j : Nat) : logical read x i j =
      VeriTile.Torch.FP32.logicalValue read x.tensor xl.tensor (i*x.columns+j) := by
  simp [logical, VeriTile.Torch.FP32.logicalValue, notNeg]

structure OutputTensor where
  object : Nat
  metadata : VeriTile.Torch.TensorMetadata
  (storageBase storageElements storageOffset : Nat)
  physicalIndex : Nat → Nat

structure OutputLaws (c : OutputTensor) (a b : Matrix) : Prop where
  dtype : c.metadata.dtype = .other "float16"
  shape : c.metadata.shape = [a.rows,b.columns]
  device : c.metadata.device = a.tensor.metadata.device
  layout : c.metadata.layout = .strided
  contiguous : c.metadata.isContiguous = Bool.true
  resolved : c.metadata.isNeg = Bool.false
  indices : ∀ i < c.metadata.numel, c.physicalIndex i = c.storageOffset+i
  inBounds : ∀ i < c.metadata.numel, c.physicalIndex i < c.storageElements
  span : c.storageBase+2*c.storageElements ≤ 2^64
  pointer : 0 < c.metadata.numel →
    c.metadata.dataPtr = BitVec.ofNat 64 (c.storageBase+2*c.storageOffset)
  alignment : 0 < c.metadata.numel → c.metadata.dataPtr.toNat%2 = 0

def outputBase (c : OutputTensor) : Nat :=
  if c.metadata.numel = 0 then 0 else c.storageBase+2*c.storageOffset
def outputAddress (c : OutputTensor) (i : Nat) : Nat :=
  c.storageBase+2*c.physicalIndex i

theorem output_numel (a b : Matrix) (c : OutputTensor) (cl : OutputLaws c a b) :
    c.metadata.numel = a.rows*b.columns := by
  simp [VeriTile.Torch.TensorMetadata.numel, cl.shape]

theorem output_address (a b : Matrix) (c : OutputTensor) (cl : OutputLaws c a b)
    (i : Nat) (hi : i < c.metadata.numel) :
    outputAddress c i = outputBase c+2*i := by
  simp [outputAddress, outputBase, show c.metadata.numel ≠ 0 by omega,
    cl.indices i hi, Nat.mul_add, Nat.add_assoc]

theorem output_span (a b : Matrix) (c : OutputTensor) (cl : OutputLaws c a b) :
    outputBase c+2*c.metadata.numel ≤ 2^64 := by
  by_cases hz : c.metadata.numel = 0
  · simp [outputBase, hz]
  have hi : c.metadata.numel-1 < c.metadata.numel := by omega
  have hb := cl.inBounds _ hi
  rw [cl.indices _ hi] at hb
  have hs := cl.span
  simp only [outputBase, hz, if_false]
  omega

theorem output_pointer (a b : Matrix) (c : OutputTensor) (cl : OutputLaws c a b)
    (hn : 0 < c.metadata.numel) : BitVec.ofNat 64 (outputBase c) = c.metadata.dataPtr := by
  simpa [outputBase, show c.metadata.numel ≠ 0 by omega] using (cl.pointer hn).symm

/-- Fresh output storage is disjoint from the complete input backing byte
intervals, including unused elements and trailing bytes. Input/input overlap
is unrestricted. These are ordinary allocation facts, not storage-ID tests. -/
structure Fresh (a b : Matrix) (c : OutputTensor) : Prop where
  a : Whole.byteSpansDisjoint (outputBase c) (2*c.metadata.numel)
    a.tensor.storage.base a.storageBytes
  b : Whole.byteSpansDisjoint (outputBase c) (2*c.metadata.numel)
    b.tensor.storage.base b.storageBytes

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly
