import bench.tritonbench_g.triton_linear_activation.Proofs.KernelValues
import bench.tritonbench_g.triton_linear_activation.Contract
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.TritonLinearActivation.KernelContract
open VeriTile.Triton FP.Structural KernelSource KernelValues
noncomputable section
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Pointer roles are output, saved scratch, flattened X, W, and optional bias.
The saved role is the actual kernel pointer, before any host copyback. -/
def parameters (r : Request) (c : Fin 98) (regions : Fin 5 → RegionName) : Parameters where
  output := regions 0
  saved := regions 1
  input := regions 2
  weight := regions 3
  bias := if r.hasBias then regions 4 else regions 2
  rows := r.batch.rows
  columns := r.n
  inner := r.k
  outputRowStride := max r.n 1
  outputColStride := 1
  savedRowStride := if r.hasSaved then max r.n 1 else 0
  savedColStride := if r.hasSaved then 1 else 0
  inputRowStride := max r.k 1
  inputInnerStride := 1
  weightColStride := max r.k 1
  weightInnerStride := 1
  blockRows := (configs c).bm
  groupRows := 8
  blockColumns := (configs c).bn
  blockInner := (configs c).bk
  hasBias := r.hasBias
  saveInputs := r.hasSaved
  activation := r.activation

structure CanonicalStrides (p : Parameters) : Prop where
  outputRow : p.outputRowStride=max p.columns 1
  outputCol : p.outputColStride=1
  savedRow : p.saveInputs=Bool.true → p.savedRowStride=max p.columns 1
  savedCol : p.saveInputs=Bool.true → p.savedColStride=1
  inputRow : p.inputRowStride=max p.inner 1
  inputInner : p.inputInnerStride=1
  weightCol : p.weightColStride=max p.inner 1
  weightInner : p.weightInnerStride=1

theorem configured_strides (r : Request) (c : Fin 98) (regions : Fin 5 → RegionName) :
    CanonicalStrides (parameters r c regions) := by
  refine ⟨rfl,rfl,?_,?_,rfl,rfl,rfl,rfl⟩
  · intro h
    change r.hasSaved=Bool.true at h
    simp only [parameters,h,if_true]
  · intro h
    change r.hasSaved=Bool.true at h
    simp only [parameters,h,if_true]

theorem configured_positive (r : Request) (c : Fin 98) (regions : Fin 5 → RegionName) :
    0<(parameters r c regions).blockInner := by
  have := config_bounds c
  change 0<(configs c).bk
  omega

/-- Active source stores are injective even when inactive ragged-lane formed
addresses collide. The guard's canonical strides suffice. -/
theorem canonical_injective {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (off : TileIndex [p.blockRows,p.blockColumns] → Nat)
    (hoff : ∀ i, off i=KernelValues.row p s i.1*max p.columns 1+KernelValues.col p s i.2.1)
    (i j : TileIndex [p.blockRows,p.blockColumns])
    (hi : active p s i=Bool.true) (hj : active p s j=Bool.true) (same : off i=off j) : i=j := by
  have boundsI : KernelValues.row p s i.1<p.rows ∧ KernelValues.col p s i.2.1<p.columns := by simpa [active] using hi
  have boundsJ : KernelValues.row p s j.1<p.rows ∧ KernelValues.col p s j.2.1<p.columns := by simpa [active] using hj
  have ci := boundsI.2
  have cj := boundsJ.2
  have npos : 0<p.columns := by omega
  simp only [hoff,max_eq_left (show 1≤p.columns by omega)] at same
  have cc : KernelValues.col p s i.2.1=KernelValues.col p s j.2.1 := by
    have modulo := congrArg (fun z => z%p.columns) same
    simpa [Nat.add_mod,Nat.mod_eq_of_lt ci,Nat.mod_eq_of_lt cj] using modulo
  have rr : KernelValues.row p s i.1=KernelValues.row p s j.1 := by
    have products := Nat.add_right_cancel (same.trans (congrArg (_+·) cc.symm))
    exact Nat.eq_of_mul_eq_mul_right npos products
  have fi : i.1=j.1 := Fin.ext (by simpa [KernelValues.row] using rr)
  have fj : i.2.1=j.2.1 := Fin.ext (by simpa [KernelValues.col] using cc)
  rcases i with ⟨i,k,⟨⟩⟩
  rcases j with ⟨j,l,⟨⟩⟩
  simp_all

theorem output_unique {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (strides : CanonicalStrides p) :
    ∀ i j, active p s i=Bool.true → active p s j=Bool.true →
      KernelValues.outputOffset p s i=KernelValues.outputOffset p s j → i=j :=
  canonical_injective p s _ (by intro i; simp [KernelValues.outputOffset,strides.outputRow,strides.outputCol])

theorem saved_unique {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (strides : CanonicalStrides p) (saved : p.saveInputs=Bool.true) :
    ∀ i j, active p s i=Bool.true → active p s j=Bool.true →
      savedOffset p s i=savedOffset p s j → i=j :=
  canonical_injective p s _ (by intro i; simp [savedOffset,strides.savedRow saved,strides.savedCol saved])

/-- Kernel storage width follows the explicit source cast branch. Ambient
bfloat16 autocast is still the FP16 branch because custom_fwd requests FP16. -/
def tensorDType : Mode → VeriTile.Torch.DType | .fp32 => .floating .fp32 | .autocastFP16 => .other "float16"
def elementBytes : Mode → Nat | .fp32 => 4 | .autocastFP16 => 2

structure HalfTensor where
  object : Nat
  metadata : VeriTile.Torch.TensorMetadata
  storageObject : Nat
  storageBase : Nat
  storageElements : Nat
  storageOffset : Nat
  physicalIndex : Nat → Nat
  spanFits : storageBase+2*storageElements≤2^64

/-- Ordinary FP16 Tensor API laws, parallel to the shared FP32 laws. These
bind addresses and metadata only; no arithmetic or output is assumed. -/
structure HalfTensorLaws (t : HalfTensor) : Prop where
  dtype : t.metadata.dtype=.other "float16"
  layout : t.metadata.layout=.strided
  inBounds : ∀ i, i<t.metadata.numel → t.physicalIndex i<t.storageElements
  pointer : 0<t.metadata.numel →
    t.metadata.dataPtr=BitVec.ofNat 64 (t.storageBase+2*t.storageOffset)
  contiguous : t.metadata.isContiguous=Bool.true → ∀ i, i<t.metadata.numel →
    t.physicalIndex i=t.storageOffset+i

/-- The FP32 case is the shared actual Tensor object and TensorLaws. The
half branch uses the same shared TensorMetadata, with its true 2-byte storage. -/
def ActualTensor : Mode → Type
  | .fp32 => {t : VeriTile.Torch.FP32.Tensor // VeriTile.Torch.FP32.TensorLaws t}
  | .autocastFP16 => {t : HalfTensor // HalfTensorLaws t}

def metadata (mode : Mode) (t : ActualTensor mode) : VeriTile.Torch.TensorMetadata :=
  match mode with | .fp32 => t.val.metadata | .autocastFP16 => t.val.metadata

def storageBase (mode : Mode) (t : ActualTensor mode) : Nat :=
  match mode with | .fp32 => t.val.storage.base | .autocastFP16 => t.val.storageBase

def storageElements (mode : Mode) (t : ActualTensor mode) : Nat :=
  match mode with | .fp32 => t.val.storage.elements | .autocastFP16 => t.val.storageElements

def storageOffset (mode : Mode) (t : ActualTensor mode) : Nat :=
  match mode with | .fp32 => t.val.storageOffset | .autocastFP16 => t.val.storageOffset

def physicalIndex (mode : Mode) (t : ActualTensor mode) : Nat → Nat :=
  match mode with | .fp32 => t.val.physicalIndex | .autocastFP16 => t.val.physicalIndex

def tensorShape (r : Request) : Fin 5 → List Nat :=
  ![[r.batch.rows,r.n],[r.batch.rows,r.n],[r.batch.rows,r.k],[r.n,r.k],[r.n]]
def used (r : Request) (slot : Fin 5) : Prop :=
  slot≠1 ∧ slot≠4 ∨ slot=1 ∧ r.hasSaved=Bool.true ∨ slot=4 ∧ r.hasBias=Bool.true

/-- Prepared pointer facts supplied at the kernel boundary. The admitted
caller remains FP32; no host autocast, allocation, or copyback theorem is added.
Complete backing separation and actual Tensor API laws are ordinary premises. -/
structure Prepared (r : Request) (regions : Fin 5 → RegionName) where
  tensors : Fin 5 → ActualTensor r.mode
  shape : ∀ slot, used r slot → (metadata r.mode (tensors slot)).shape=tensorShape r slot
  contiguous : ∀ slot, used r slot → (metadata r.mode (tensors slot)).isContiguous=Bool.true
  offsetZero : ∀ slot, used r slot → storageOffset r.mode (tensors slot)=0
  noNeg : ∀ slot, used r slot → (metadata r.mode (tensors slot)).isNeg=Bool.false
  cuda : ∀ slot, used r slot → (metadata r.mode (tensors slot)).device=⟨"cuda",r.device⟩
  aligned : ∀ slot, used r slot → 0<(metadata r.mode (tensors slot)).numel →
    (metadata r.mode (tensors slot)).dataPtr.toNat%elementBytes r.mode=0
  regionDistinct : Function.Injective regions
  separate : ∀ i j, used r i → used r j → i≠j →
    0<(metadata r.mode (tensors i)).numel → 0<(metadata r.mode (tensors j)).numel →
    storageBase r.mode (tensors i)+elementBytes r.mode*storageElements r.mode (tensors i)≤storageBase r.mode (tensors j) ∨
    storageBase r.mode (tensors j)+elementBytes r.mode*storageElements r.mode (tensors j)≤storageBase r.mode (tensors i)

theorem physical_contiguous (mode : Mode) (t : ActualTensor mode)
    (contiguous : (metadata mode t).isContiguous=Bool.true)
    (offset : storageOffset mode t=0) (i : Nat) (bound : i<(metadata mode t).numel) :
    physicalIndex mode t i=i := by
  cases mode
  · change t.val.storageOffset=0 at offset
    simpa [physicalIndex,offset] using t.property.contiguous contiguous i bound
  · change t.val.storageOffset=0 at offset
    simpa [physicalIndex,offset] using t.property.contiguous contiguous i bound

theorem capacity (mode : Mode) (t : ActualTensor mode)
    (contiguous : (metadata mode t).isContiguous=Bool.true)
    (offset : storageOffset mode t=0) (i : Nat) (bound : i<(metadata mode t).numel) :
    i<storageElements mode t := by
  have physical := physical_contiguous mode t contiguous offset i bound
  cases mode
  · change t.val.physicalIndex i=i at physical
    simpa [storageElements,physical] using (t.property.inBounds i bound)
  · change t.val.physicalIndex i=i at physical
    simpa [storageElements,physical] using (t.property.inBounds i bound)

#axiomsClean output_unique
#axiomsClean saved_unique
#axiomsClean physical_contiguous
#axiomsClean capacity
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonLinearActivation.KernelContract
