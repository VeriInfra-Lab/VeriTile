import bench.repairs.tritonbench_g.triton_matmul.Proofs.Contract
import bench.tritonbench_g.triton_matmul.Proofs.FPReference
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.TensorBinding
open VeriTile.Triton FP.Structural FPValue VeriTile.Torch
open Contract
noncomputable section

abbrev Word := FP.Value .fp32
abbrev Read := Nat → BitVec 8
def storage (t : TensorMetadata) : Storage := if t.dtype=.other "float16" then .half else .e4m3
def torchType : Storage → DType | .half => .other "float16" | .e4m3 => .other "float8_e4m3fn"
def width : Storage → Nat | .half => 2 | .e4m3 => 1
def sign : Storage → Nat | .half => 32768 | .e4m3 => 128

structure Tensor where
  object : Nat
  metadata : TensorMetadata
  (storageId storageBase storageElements storageOffset : Nat)
  physicalIndex : Nat → Nat

/-- Ordinary Torch storage/view semantics. Kernel-specific index and byte
envelopes are absent; the checked input contract supplies them. -/
structure Laws (st : Storage) (t : Tensor) : Prop where
  dtype : t.metadata.dtype=torchType st
  layout : t.metadata.layout=.strided
  bounds : ∀ i,i<t.metadata.numel → t.physicalIndex i<t.storageElements
  pointer : 0<t.metadata.numel → t.metadata.dataPtr.toNat=t.storageBase+width st*t.storageOffset
  contiguous : t.metadata.isContiguous=Bool.true →
    ∀ i,i<t.metadata.numel → t.physicalIndex i=t.storageOffset+i

/-- Same packed FP16/E4M3FN carrier used by the existing typed source. -/
def readWord (read : Read) (st : Storage) (address : Nat) : Word := ⟨BitVec.ofNat 32
  (match st with
   | .half => (read address).toNat+256*(read (address+1)).toNat
   | .e4m3 => (read address).toNat)⟩
def physicalAddress (st : Storage) (t : Tensor) (i : Nat) := t.storageBase+width st*t.physicalIndex i
def logicalValue (read : Read) (st : Storage) (t : Tensor) (_laws : Laws st t) (i : Nat) : Word :=
  let raw := readWord read st (physicalAddress st t i)
  if t.metadata.isNeg then ⟨raw.bits ^^^ BitVec.ofNat FP.Format.fp32.width (sign st)⟩ else raw

theorem checked_dtype (t : TensorMetadata) (h : Supported t) : t.dtype=torchType (storage t) := by
  rcases h with h | h <;> simp [storage,torchType,h]
theorem checked_bytes (t : TensorMetadata) : bytes t=width (storage t) := by
  by_cases h : t.dtype=.other "float16" <;> simp [bytes,width,storage,h]
theorem same_storage (a b : TensorMetadata) (h : b.dtype=a.dtype) : storage b=storage a := by
  simp only [storage,h]

theorem direct_read (read : Read) (st : Storage) (t : Tensor) (laws : Laws st t)
    (contiguous : t.metadata.isContiguous=Bool.true) (resolved : t.metadata.isNeg=Bool.false)
    (i : Nat) (hi : i<t.metadata.numel) :
    readWord read st (t.metadata.dataPtr.toNat+width st*i)=logicalValue read st t laws i := by
  rw [laws.pointer (by omega)]
  simp only [logicalValue,resolved,Bool.false_eq_true,↓reduceIte,physicalAddress,
    laws.contiguous contiguous i hi,Nat.mul_add,Nat.add_assoc]

def inputA : RegionName := "A"
def inputB : RegionName := "B"
def outputC : RegionName := "C"

/-- Both original pointer views project the same physical byte reader.
Consequently overlapping read-only tensors retain coherent logical values. -/
structure Inputs {α : Type} (read : Read) (decode : Word → α)
    (a b : Tensor) (s : State α) : Prop where
  left : ∀ i,i<a.metadata.numel → s.mem inputA i=
    stored (storage a.metadata) (decode (readWord read (storage a.metadata)
      (a.metadata.dataPtr.toNat+width (storage a.metadata)*i)))
  right : ∀ i,i<b.metadata.numel → s.mem inputB i=
    stored (storage b.metadata) (decode (readWord read (storage b.metadata)
      (b.metadata.dataPtr.toNat+width (storage b.metadata)*i)))

theorem linear_bound (r c M N : Nat) (hr : r<M) (hc : c<N) : r*N+c<M*N := by
  have h := Nat.mul_le_mul_right N (show r+1≤M by omega)
  nlinarith

theorem read_stored {α : Type} [Inhabited α] (st : Storage) (x : α) :
    FPValue.read st (stored st x)=x := by cases st <;> rfl

theorem logical_left {α : Type} [Inhabited α] (read : Read) (decode : Word → α)
    (a b : Tensor) (la : Laws (storage a.metadata) a) (h : InputContract a.metadata b.metadata)
    (s : State α) (bound : Inputs read decode a b s) (i : Nat) (hi : i<a.metadata.numel) :
    FPValue.read (storage a.metadata) (s.mem inputA i)=decode (logicalValue read (storage a.metadata) a la i) := by
  rw [bound.left i hi,direct_read read _ a la h.contiguous.1 h.resolved.1 i hi]
  exact read_stored _ _

theorem logical_right {α : Type} [Inhabited α] (read : Read) (decode : Word → α)
    (a b : Tensor) (lb : Laws (storage b.metadata) b) (h : InputContract a.metadata b.metadata)
    (s : State α) (bound : Inputs read decode a b s) (i : Nat) (hi : i<b.metadata.numel) :
    FPValue.read (storage a.metadata) (s.mem inputB i)=decode (logicalValue read (storage b.metadata) b lb i) := by
  rw [bound.right i hi,direct_read read _ b lb h.contiguous.2 h.resolved.2 i hi,←same_storage a.metadata b.metadata h.dtype.2]
  exact read_stored _ _

#axiomsClean direct_read
#axiomsClean logical_left
#axiomsClean logical_right
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.TensorBinding
