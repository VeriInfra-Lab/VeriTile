import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.KernelSource
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.CacheTransform.KernelBinding
open VeriTile.Triton FP.Structural VeriTile.Torch
noncomputable section
open Classical
set_option maxRecDepth 10000

abbrev Byte := BitVec 8
abbrev Bytes := Nat → Byte

/-- Ordinary little-endian FP32 word interpretation of four actual bytes.
No floating arithmetic or finite-value restriction is involved. -/
def assemble32 (bytes : Fin 4 → Nat) : VeriTile.Torch.FP32.Word :=
  ⟨BitVec.ofNat 32 (∑ b, bytes b * 256^b.val)⟩
def read32 (memory : Bytes) (address : Nat) : VeriTile.Torch.FP32.Word :=
  assemble32 (fun b => (memory (address+b.val)).toNat)
def read64 (memory : Bytes) (address : Nat) : BitVec 64 :=
  BitVec.ofNat 64 (∑ b : Fin 8, (memory (address+b.val)).toNat * 256^b.val)

structure Cache (metadata : TensorMetadata) where
  tensor : VeriTile.Torch.FP32.Tensor
  descriptor : tensor.metadata=metadata
  laws : VeriTile.Torch.FP32.TensorLaws tensor

def Cache.view (x : Cache metadata) (contiguous : metadata.isContiguous=Bool.true) :
    VeriTile.Torch.FP32.DataView :=
  VeriTile.Torch.FP32.dataView x.tensor x.laws (by simpa only [x.descriptor] using contiguous)

def Cache.real (x : Cache metadata) (memory : Nat → ℝ) (i : Nat) : ℝ :=
  memory (x.tensor.storage.address (x.tensor.physicalIndex i))
def Cache.word (x : Cache metadata) (memory : Bytes) (i : Nat) : VeriTile.Torch.FP32.Word :=
  VeriTile.Torch.FP32.logicalValue (read32 memory) x.tensor x.laws i

theorem Cache.address (x : Cache metadata) (contiguous : metadata.isContiguous=Bool.true)
    (i : Nat) (inside : i<metadata.numel) :
    (x.view contiguous).address i=x.tensor.storage.address (x.tensor.physicalIndex i) := by
  have bound : i<x.tensor.metadata.numel := by simpa only [x.descriptor] using inside
  rw [x.laws.contiguous (by simpa only [x.descriptor] using contiguous) i bound]
  simp only [Cache.view,VeriTile.Torch.FP32.dataView,VeriTile.Torch.FP32.DataView.address,
    VeriTile.Torch.FP32.DataView.base,if_neg (by omega : x.tensor.metadata.numel≠0),
    VeriTile.Torch.FP32.Storage.address]
  omega

theorem Cache.pointer (x : Cache metadata) (contiguous : metadata.isContiguous=Bool.true)
    (offset : Nat) (inside : offset<4*metadata.numel) :
    (metadata.dataPtr+BitVec.ofNat 64 offset).toNat=(x.view contiguous).base+offset := by
  have nonempty : 0<x.tensor.metadata.numel := by rw [x.descriptor]; omega
  have pointer := VeriTile.Torch.FP32.actual_pointer x.tensor x.laws
    (by simpa only [x.descriptor] using contiguous) nonempty
  change BitVec.ofNat 64 ((x.view contiguous).address 0)=x.tensor.metadata.dataPtr at pointer
  rw [x.descriptor] at pointer
  rw [←pointer,BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat]
  have span := (x.view contiguous).spanFits
  have elements : (x.view contiguous).elements=metadata.numel := by
    simp only [Cache.view,VeriTile.Torch.FP32.dataView,x.descriptor]
  rw [elements] at span
  simp only [VeriTile.Torch.FP32.DataView.address,Nat.mul_zero,Nat.add_zero]
  exact Nat.mod_eq_of_lt (by omega)

theorem Cache.word_direct (x : Cache metadata) (contiguous : metadata.isContiguous=Bool.true)
    (noNeg : metadata.isNeg=Bool.false) (memory : Bytes) (i : Nat) (inside : i<metadata.numel) :
    read32 memory ((x.view contiguous).address i)=x.word memory i := by
  apply VeriTile.Torch.FP32.direct_read_eq_logical
  · simpa only [x.descriptor] using noNeg
  · simpa only [x.descriptor] using inside

/-- Actual strided int32 Tensor observations used by cpu().tolist(). Cache
contiguity is not imposed on lengths, and its lazy negation is interpreted. -/
structure LengthTensor (metadata : TensorMetadata) where
  (base offset stride elements : Nat)
  span : base+4*elements≤2^64
  pointer : 0<metadata.numel → metadata.dataPtr=BitVec.ofNat 64 (base+4*offset)
  bound : ∀ i,i<metadata.numel → offset+i*stride<elements

def LengthTensor.value (x : LengthTensor metadata) (memory : Bytes) (i : Nat) : Int :=
  let raw := (read32 memory (x.base+4*(x.offset+i*x.stride))).bits
  (if metadata.isNeg then -raw else raw).toInt

structure Inputs (r : Request) (memory : Bytes) where
  cos : Cache r.cos
  sin : Cache r.sin
  lengths : LengthTensor r.lengths
  values : ∀ i : Fin r.values.length, r.values[i]=lengths.value memory i.val

/-- Actual constructor result for torch.tensor(rows,dtype=int64). -/
structure Table (r : Request) (memory : Bytes) where
  metadata : TensorMetadata
  shape : 0<r.count → metadata.shape=[r.outputRows]
  dtype : 0<r.count → metadata.dtype=.other "int64"
  layout : 0<r.count → metadata.layout=.strided
  device : 0<r.count → metadata.device=r.cos.device
  contiguous : 0<r.count → metadata.isContiguous=Bool.true
  /-- Ordinary int64 constructor alignment, on the same observed pointer. -/
  aligned : 0<r.count → metadata.dataPtr.toNat%8=0
  span : 0<r.count → metadata.dataPtr.toNat+8*r.outputRows≤2^64
  values : 0<r.count → ∀ q,q<r.outputRows →
    (read64 memory (metadata.dataPtr.toNat+8*q)).toInt=(tableRows r.mode r.naturalLengths q : Int)

structure Output (r : Request) where
  tensor : VeriTile.Torch.FP32.Tensor
  laws : VeriTile.Torch.FP32.TensorLaws tensor
  shape : tensor.metadata.shape=[r.outputRows,r.width]
  device : tensor.metadata.device=r.cos.device
  contiguous : tensor.metadata.isContiguous=Bool.true
  noNeg : tensor.metadata.isNeg=Bool.false

def Output.cache (z : Output r) : Cache z.tensor.metadata := ⟨z.tensor,rfl,z.laws⟩
def Output.view (z : Output r) := z.cache.view z.contiguous

theorem Output.size (z : Output r) : z.tensor.metadata.numel=r.outputRows*r.width := by
  simp [TensorMetadata.numel,z.shape]

def inside (base count address : Nat) : Prop := base≤address ∧ address<base+count

def Output.contains (z : Output r) (address : Nat) : Prop := inside z.view.base r.count address

/-- Fresh outputs are actual FP32 Tensor objects with observed pointers and
coordinates. Separation is an allocator API fact, never an input-value guard. -/
structure Outputs (r : Request) where
  cos : Output r
  sin : Output r
  separate : ∀ address,cos.contains address → ¬sin.contains address
  caller : Nat → Prop
  freshCos : ∀ address,caller address → ¬cos.contains address
  freshSin : ∀ address,caller address → ¬sin.contains address

/-- Byte or mathematical-value realization of the complete source's two
written windows at their actual Tensor pointers. Every byte in those windows
is written by the proved kernel grid; exterior addresses retain their values. -/
def realize (z : Outputs r) (before : Nat → α) (cos sin : Nat → α) (address : Nat) : α :=
  if z.cos.contains address then cos (address-z.cos.view.base)
  else if z.sin.contains address then sin (address-z.sin.view.base) else before address

theorem realize_cos (z : Outputs r) (before cos sin : Nat → α) (o : Nat) (ho : o<r.count) :
    realize z before cos sin (z.cos.view.base+o)=cos o := by
  have inside : z.cos.contains (z.cos.view.base+o) := by
    simp only [Output.contains,inside]; omega
  simp [realize,inside]

theorem realize_sin (z : Outputs r) (before cos sin : Nat → α) (o : Nat) (ho : o<r.count) :
    realize z before cos sin (z.sin.view.base+o)=sin o := by
  have inside : z.sin.contains (z.sin.view.base+o) := by
    simp only [Output.contains,inside]; omega
  have outside : ¬z.cos.contains (z.sin.view.base+o) := fun hc => z.separate _ hc inside
  simp [realize,outside,inside]

theorem realize_frame (z : Outputs r) (before cos sin : Nat → α) (address : Nat)
    (hc : ¬z.cos.contains address) (hs : ¬z.sin.contains address) :
    realize z before cos sin address=before address := by simp [realize,hc,hs]

theorem realize_caller (z : Outputs r) (before cos sin : Nat → α) (address : Nat)
    (caller : z.caller address) : realize z before cos sin address=before address :=
  realize_frame z before cos sin address (z.freshCos address caller) (z.freshSin address caller)

theorem cos_contiguous (h : InputContract r) : r.cos.isContiguous=Bool.true := by
  rcases h.1 with ⟨_,_,_,_,_,_,_,_,_,_,_,_,hc,_,_,_,_,_⟩; exact hc
theorem sin_contiguous (h : InputContract r) : r.sin.isContiguous=Bool.true := by
  rcases h.1 with ⟨_,_,_,_,_,_,_,_,_,_,_,_,_,hs,_,_,_,_⟩; exact hs
theorem cos_noNeg (h : InputContract r) : r.cos.isNeg=Bool.false := by
  rcases h.1 with ⟨_,_,_,_,_,_,_,_,_,_,_,_,_,_,hc,_,_,_⟩; exact hc
theorem sin_noNeg (h : InputContract r) : r.sin.isNeg=Bool.false := by
  rcases h.1 with ⟨_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,hs,_,_⟩; exact hs
theorem cos_shape (h : InputContract r) : r.cos.shape=[r.cosRows,r.width] := by
  rcases h.1 with ⟨_,_,_,_,_,_,_,hc,_,_,_,_,_,_,_,_,_,_⟩; exact hc
theorem sin_shape (h : InputContract r) : r.sin.shape=[r.sinRows,r.width] := by
  rcases h.1 with ⟨_,_,_,_,_,_,_,_,hs,_,_,_,_,_,_,_,_,_⟩; exact hs

def Inputs.cosBytes (x : Inputs r memory) (h : InputContract r) (o : Nat) : Nat :=
  (memory ((x.cos.view (cos_contiguous h)).base+o)).toNat
def Inputs.sinBytes (x : Inputs r memory) (h : InputContract r) (o : Nat) : Nat :=
  (memory ((x.sin.view (sin_contiguous h)).base+o)).toNat

def Table.read (table : Table r memory) (q : Nat) : Int :=
  (read64 memory (table.metadata.dataPtr.toNat+8*q)).toInt

/-- The caller predicate includes the actual cache windows, strided lengths'
backing allocation and live row table. Outputs may not overwrite these bytes. -/
structure Allocation (x : Inputs r memory) (h : InputContract r)
    (cosTable sinTable : Table r memory) (z : Outputs r) : Prop where
  cos : ∀ o,o<4*r.cos.numel → z.caller ((x.cos.view (cos_contiguous h)).base+o)
  sin : ∀ o,o<4*r.sin.numel → z.caller ((x.sin.view (sin_contiguous h)).base+o)
  lengths : ∀ o,o<4*x.lengths.elements → z.caller (x.lengths.base+o)
  cosTable : 0<r.count → ∀ o,o<8*r.outputRows → z.caller (cosTable.metadata.dataPtr.toNat+o)
  sinTable : 0<r.count → ∀ o,o<8*r.outputRows → z.caller (sinTable.metadata.dataPtr.toNat+o)

/-- Read a result by its actual output Tensor's logical coordinates. -/
def realTensor (z : Output r) (memory : Nat → ℝ) : List (Fin r.width → ℝ) :=
  List.ofFn (fun row : Fin r.outputRows => fun col =>
    z.cache.real memory (row.val*r.width+col.val))
def wordTensor (z : Output r) (memory : Bytes) : List (Fin r.width → VeriTile.Torch.FP32.Word) :=
  List.ofFn (fun row : Fin r.outputRows => fun col =>
    z.cache.word memory (row.val*r.width+col.val))

#axiomsClean Cache.address
#axiomsClean Cache.pointer
#axiomsClean Cache.word_direct
#axiomsClean realize_cos
#axiomsClean realize_sin
#axiomsClean realize_frame
#axiomsClean realize_caller
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CacheTransform.KernelBinding
