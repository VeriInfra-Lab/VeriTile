import bench.tritonbench_g.ksoftmax_triton.Proofs.Binding
import bench.tritonbench_g.ksoftmax_triton.Proofs.Pointers

/-! Ordinary immutable Tensor views. Contiguity concerns address metadata,
never kernel outputs. Empty tensors impose no pointer/alignment requirement.
The original storage base and offset and harmless singleton strides remain.
-/
namespace VeriTile.Bench.Repairs.Ksoftmax.KernelTensor
open VeriTile.Triton FP.Structural
noncomputable section

def coordinates : List Nat → Nat → List Nat
  | [],_ => []
  | n::ns,j => j/ns.prod :: coordinates ns (j%ns.prod)

structure View (metadata : TensorMetadata) where
  storageBase : Nat
  storageBytes : Nat
  storageOffset : Nat
  strides : List Nat
  rank : strides.length=metadata.shape.length
  pointer : 0<metadata.numel → metadata.dataPtr.toNat=storageBase+4*storageOffset
  backing : storageBase+storageBytes≤2^64
  span : 0<metadata.numel → 4*storageOffset+4*metadata.numel≤storageBytes
  /-- Truthful default-format contiguity, permitting arbitrary singleton strides. -/
  contiguous : metadata.isContiguous=Bool.true → ∀ j, j<metadata.numel →
    (List.zipWith (· * ·) (coordinates metadata.shape j) strides).sum=j

def address (metadata : TensorMetadata) (view : View metadata) (j : Nat) : Nat :=
  view.storageBase+4*(view.storageOffset+
    (List.zipWith (· * ·) (coordinates metadata.shape j) view.strides).sum)

theorem address_flat (metadata : TensorMetadata) (view : View metadata)
    (admitted : TensorContract metadata) (j : Nat) (hj : j<metadata.numel) :
    address metadata view j=metadata.dataPtr.toNat+4*j := by
  rw [address,view.contiguous admitted.2.2.2.1 j hj,view.pointer (by omega)]
  omega

theorem span64 (metadata : TensorMetadata) (view : View metadata) (live : 0<metadata.numel) :
    metadata.dataPtr.toNat+4*metadata.numel≤2^64 := by
  rw [view.pointer live]
  have h := view.span live
  have b := view.backing
  omega

def words (memory : Nat → Graph.Word) (metadata : TensorMetadata) (j : Nat) : Graph.Word :=
  memory (metadata.dataPtr.toNat+4*j)

/-- The ordinary initial FP32 word/four-byte storage binding for every active
Tensor role. Read aliases therefore cannot carry inconsistent initial bytes.
This says nothing about any output, execution, arithmetic, or final memory. -/
def Backed (memory : Nat → Graph.Word) (before : Binding.Bytes) (roles : List TensorMetadata) : Prop :=
  ∀ metadata ∈ roles, ∀ j, j<metadata.numel → ∀ byte, byte<4 →
    before (metadata.dataPtr.toNat+4*j+byte)=Binding.byteOf (words memory metadata j) byte

def logical (memory : Nat → Graph.Word) (metadata : TensorMetadata) (view : View metadata)
    (j : Nat) : Graph.Word := memory (address metadata view j)

theorem words_logical (memory : Nat → Graph.Word) (metadata : TensorMetadata)
    (view : View metadata) (admitted : TensorContract metadata) (j : Nat) (hj : j<metadata.numel) :
    words memory metadata j=logical memory metadata view j := by
  simp only [words,logical,address_flat metadata view admitted j hj]

/-- The logical tensor sequence, with an irrelevant extension outside its shape. -/
def values (memory : Nat → Graph.Word) (metadata : TensorMetadata) (view : View metadata)
    (j : Nat) : Graph.Word :=
  if j<metadata.numel then logical memory metadata view j else words memory metadata j

theorem values_words (memory : Nat → Graph.Word) (metadata : TensorMetadata)
    (view : View metadata) (admitted : TensorContract metadata) :
    values memory metadata view=words memory metadata := by
  funext j
  unfold values
  split
  · exact (words_logical memory metadata view admitted j ‹_›).symm
  · rfl

/-- All aliases read the same physical word map. Source memory roles are tags,
not an assumption that the caller's read tensors occupy disjoint allocations. -/
def initial {α : Type} (interpret : Nat → Nat → Graph.Word → α)
    (dst left right : TensorMetadata) (memory : Nat → Graph.Word) (s : State α) : State α :=
  {s with mem := fun region j =>
    if region="dst" then Cell.mk .real (interpret 2 j (words memory dst j))
    else if region="left" then Cell.mk .real (interpret 0 j (words memory left j))
    else if region="right" then Cell.mk .real (interpret 1 j (words memory right j))
    else s.mem region j}

def nativeInitial (dst left right : TensorMetadata) (memory : Nat → Graph.Word)
    (s : State Graph.Expr) : State Graph.Expr :=
  initial (fun role j word => if role=2 then .leaf word else .input role j word) dst left right memory s

def idealInitial (projection : Graph.Word → WithBot ℝ) (dst left right : TensorMetadata)
    (memory : Nat → Graph.Word) (s : State (WithBot ℝ)) : State (WithBot ℝ) :=
  initial (fun _ _ word => projection word) dst left right memory s

theorem native_left (dst left right : TensorMetadata) (memory : Nat → Graph.Word)
    (s : State Graph.Expr) (j : Nat) :
    ((nativeInitial dst left right memory s).mem "left" j).read .real=
      .input 0 j (words memory left j) := by simp [nativeInitial,initial]

theorem native_right (dst left right : TensorMetadata) (memory : Nat → Graph.Word)
    (s : State Graph.Expr) (j : Nat) :
    ((nativeInitial dst left right memory s).mem "right" j).read .real=
      .input 1 j (words memory right j) := by simp [nativeInitial,initial]

theorem native_dst (dst left right : TensorMetadata) (memory : Nat → Graph.Word)
    (s : State Graph.Expr) (j : Nat) :
    ((nativeInitial dst left right memory s).mem "dst" j).read .real=
      .leaf (words memory dst j) := by simp [nativeInitial,initial]

theorem ideal_left (projection : Graph.Word → WithBot ℝ) (dst left right : TensorMetadata)
    (memory : Nat → Graph.Word) (s : State (WithBot ℝ)) (j : Nat) :
    ((idealInitial projection dst left right memory s).mem "left" j).read .real=
      projection (words memory left j) := by simp [idealInitial,initial]

theorem ideal_right (projection : Graph.Word → WithBot ℝ) (dst left right : TensorMetadata)
    (memory : Nat → Graph.Word) (s : State (WithBot ℝ)) (j : Nat) :
    ((idealInitial projection dst left right memory s).mem "right" j).read .real=
      projection (words memory right j) := by simp [idealInitial,initial]

/-- Logical output coordinates are observed through the same actual Tensor
strides, backing offset and data pointer, including all four FP32 bytes. -/
theorem output_bytes (ops : Graph.Operations) (dst : TensorMetadata) (view : View dst)
    (admitted : TensorContract dst) (before : Binding.Bytes) (s : State Graph.Expr)
    (j byte : Nat) (hj : j<dst.numel) (hb : byte<4) :
    Binding.callerBytes ops dst before s (address dst view j+byte)=
      Binding.byteOf (Graph.eval ops ((s.mem "dst" j).read .real)) byte := by
  rw [address_flat dst view admitted j hj]
  exact Binding.coordinate_bytes ops dst before s j byte hj hb

#specNonCircular View avoiding [runRows,Host.body]
#specNonCircular Backed avoiding [runRows,Host.body,Graph.eval]
#specNonCircular initial avoiding [runRows,Host.body]
#axiomsClean address_flat
#axiomsClean span64
#axiomsClean words_logical
#axiomsClean values_words
#axiomsClean native_left
#axiomsClean native_right
#axiomsClean native_dst
#axiomsClean ideal_left
#axiomsClean ideal_right
#axiomsClean output_bytes
end
end VeriTile.Bench.Repairs.Ksoftmax.KernelTensor
