import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Native
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Math

/-! Kernel argument tensors, pointers, and initial storage. Logical regions
name typed pointer views; views may share an actual allocation, whose byte
coherence is recorded below. Kernel execution is proved in KernelNative. -/
namespace VeriTile.Bench.Repairs.CrossEntropy1.KernelContract
open VeriTile.Triton FP.Structural
open TensorModel (Tensor coordinateOffset)
noncomputable section

abbrev Word := BitVec 32
abbrev Byte := BitVec 8

/-- Physical backing of a typed pointer view, in bytes. Distinct logical
regions may have identical backing; tensor offsets are in dtype elements. -/
abbrev Allocation := TensorModel.Native.Buffer

def fpValue {α : Type} [Inhabited α] (initial : State α) (t : Tensor) (i : Nat) : α :=
  (initial.mem t.region (t.index i)).read .real

def targetValue {α : Type} [Inhabited α] (initial : State α) (t : Tensor) (i : Nat) : Int :=
  (initial.mem t.region (t.index i)).read .int

/-- Little-endian bits of an actual storage element. -/
def wordAt (bytes : Nat → Byte) (offset : Nat) : Word :=
  BitVec.ofNat 32 (∑ j : Fin 4, (bytes (offset+j.val)).toNat * 256^j.val)

def intAt (bytes : Nat → Byte) (offset : Nat) : Int :=
  ((wordAt bytes (offset+4)).append (wordAt bytes offset)).toInt

/-- Aliased logical views observe the same physical byte at the same address. -/
def CoherentBytes (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte) : Prop :=
  ∀ r s i j, (alloc r).device=(alloc s).device →
    i<(alloc r).capacity → j<(alloc s).capacity → (alloc r).base+i=(alloc s).base+j → bytes r i=bytes s j

/-- Ordinary pointer/backing contract. Only logical tensor elements must be
backed; neither zero offsets nor a multiple-of-element-width allocation size is
required. This is an input contract, with no kernel-result field. -/
structure TensorStorage (alloc : RegionName → Allocation) (t : Tensor)
    (elementBytes pointer : Nat) : Prop where
  cuda : t.isCUDA=Bool.true
  noNegative : t.isNeg=Bool.false
  device : t.device=(alloc t.region).device
  rank : t.strides.length=t.shape.length
  pointerEq : pointer=(alloc t.region).base+elementBytes*t.storageOffset
  aligned : 0<t.numel → pointer%elementBytes=0
  addressSpace : (alloc t.region).base+(alloc t.region).capacity≤2^64
  backed : ∀ i, i<t.numel → elementBytes*(t.index i+1)≤(alloc t.region).capacity

/-- The native FP32 view reads its exact four backing bytes. -/
structure FloatBinding (alloc : RegionName → Allocation)
    (bytes : RegionName → Nat → Byte) (initial : State Word) (t : Tensor)
    (pointer : Nat) : Prop extends TensorStorage alloc t 4 pointer where
  dtype : t.dtype="torch.float32"
  cells : ∀ i, i<t.numel → fpValue initial t i=wordAt (bytes t.region) (4*t.index i)

/-- Integer labels are linked to the same signed int64 backing bytes used by
the source label load; an unrelated target function cannot satisfy this field. -/
structure LabelBinding (alloc : RegionName → Allocation)
    (bytes : RegionName → Nat → Byte) (initial : State Word) (t : Tensor)
    (pointer : Nat) : Prop extends TensorStorage alloc t 8 pointer where
  dtype : t.dtype="torch.int64"
  cells : ∀ i, i<t.numel → targetValue initial t i=intAt (bytes t.region) (8*t.index i)

structure MatrixView (t : Tensor) (region : RegionName) (base rows cols stride : Nat) : Prop where
  regionEq : t.region=region
  shape : t.shape=[rows,cols]
  strides : t.strides=[stride,1]
  offset : t.storageOffset=base

structure VectorView (t : Tensor) (region : RegionName) (base count stride : Nat) : Prop where
  regionEq : t.region=region
  shape : t.shape=[count]
  strides : t.strides=[stride]
  offset : t.storageOffset=base

/-- The selected caller uses `[rows]` for one block and `[blocks,rows]` for
multiple blocks. Both pass the same flat block-major pointer layout. -/
structure ScalarOutputView (t : Tensor) (region : RegionName) (base rows blocks : Nat) : Prop where
  regionEq : t.region=region
  shape : t.shape=if 1<blocks then [blocks,rows] else [rows]
  strides : t.strides=if 1<blocks then [rows,1] else [1]
  offset : t.storageOffset=base

theorem matrix_index (t : Tensor) (region : RegionName) (base rows cols stride row col : Nat)
    (view : MatrixView t region base rows cols stride) (hr : row<rows) (hc : col<cols) :
    t.index (row*cols+col)=base+row*stride+col := by
  have quotient : (row*cols+col)/cols=row := by
    apply Nat.div_eq_of_lt_le
    · omega
    · rw [Nat.add_mul,Nat.one_mul]
      omega
  have remainder : (row*cols+col)%cols=col := by
    have h := Nat.div_add_mod (row*cols+col) cols
    rw [quotient,Nat.mul_comm cols row] at h
    omega
  simp [Tensor.index,coordinateOffset,view.shape,view.strides,view.offset,
    quotient,remainder,Nat.mod_eq_of_lt hr,Nat.mod_eq_of_lt hc,Nat.add_assoc]

theorem vector_index (t : Tensor) (region : RegionName) (base count stride i : Nat)
    (view : VectorView t region base count stride) (hi : i<count) :
    t.index i=base+i*stride := by
  simp [Tensor.index,coordinateOffset,view.shape,view.strides,view.offset,Nat.mod_eq_of_lt hi]

theorem output_index (t : Tensor) (region : RegionName) (base rows blocks row block : Nat)
    (view : ScalarOutputView t region base rows blocks) (hr : row<rows) (hb : block<blocks) :
    t.index (block*rows+row)=base+block*rows+row := by
  by_cases many : 1<blocks
  · exact matrix_index t region base blocks rows rows block row
      ⟨view.regionEq,by simpa only [if_pos many] using view.shape,
        by simpa only [if_pos many] using view.strides,view.offset⟩ hb hr
  · have zero : block=0 := by omega
    have result := vector_index t region base rows 1 row
      ⟨view.regionEq,by simpa only [if_neg many] using view.shape,
        by simpa only [if_neg many] using view.strides,view.offset⟩ hr
    simpa only [zero,Nat.zero_mul,Nat.add_zero,Nat.zero_add,Nat.mul_one] using result

structure Launch (p : Parameters) where
  row : Nat
  block : Nat
  rowBound : row<p.rows
  blockBound : block<Math.blocks p.cols p.block

def LaunchState {α : Type} (p : Parameters) (launch : Launch p) (initial : State α) : Prop :=
  initial.pids 0=launch.row ∧ initial.pids 1=launch.block

/-- Kernel launch and scalar ABI, independent of the host implementation.
The Python checks imply the selected single-process specialization. -/
structure ParametersValid (p : Parameters) : Prop where
  geometry : Native.Supported p.rows (Math.blocks p.cols p.block) p.block p.cols
  powerOfTwo : ∃ k, p.block=2^k
  classesPositive : 0<p.cols
  totalClasses : p.totalClasses=p.cols
  classStart : p.classStart=0
  smoothingRange : 0≤p.smoothing ∧ p.smoothing≤1
  scaleNonnegative : 0≤p.squareScale
  smoothingFlag : p.hasSmoothing=decide (0<p.smoothing)
  ignoredRange : -(2^63 : Int)≤p.ignored ∧ p.ignored<2^63
  stridesRepresentable : p.logitsStride<2^63 ∧ p.gradientStride<2^63 ∧ p.upstreamStride<2^63

/-- These are views of arguments already supplied to the JIT. No allocation,
copy, autograd, aggregation or wrapper execution is included. -/
structure ForwardViews (p : Parameters) where
  logits : Tensor
  labels : Tensor
  loss : Tensor
  lse : Tensor
  logitsView : MatrixView logits p.logits p.logitsBase p.rows p.cols p.logitsStride
  labelsView : VectorView labels (Region.cast p.labels) p.labelsBase p.rows 1
  lossView : ScalarOutputView loss p.loss p.lossBase p.rows (Math.blocks p.cols p.block)
  lseView : ScalarOutputView lse p.lse p.lseBase p.rows (Math.blocks p.cols p.block)
  distinctOutputs : p.lse≠p.loss
  lseLogits : p.lse≠p.logits

structure BackwardViews (p : Parameters) where
  logits : Tensor
  labels : Tensor
  lse : Tensor
  upstream : Tensor
  gradient : Tensor
  logitsView : MatrixView logits p.logits p.logitsBase p.rows p.cols p.logitsStride
  labelsView : VectorView labels (Region.cast p.labels) p.labelsBase p.rows 1
  lseView : VectorView lse p.lse p.lseBase p.rows 1
  upstreamView : VectorView upstream p.upstream p.upstreamBase p.rows p.upstreamStride
  gradientView : MatrixView gradient p.gradient p.gradientBase p.rows p.cols p.gradientStride

structure ForwardInputs (p : Parameters) (views : ForwardViews p)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte)
    (initial : State Word) where
  parameters : ParametersValid p
  scalars : RuntimeArgs.Values Word
  scalarBinding : RuntimeArgs.Bound p initial scalars
  byteCoherence : CoherentBytes alloc bytes
  typedViews : p.logits≠Region.cast p.labels
  splitCoverage : p.split=Bool.false → p.cols≤p.block
  logitsPointer : Nat
  labelsPointer : Nat
  lossPointer : Nat
  lsePointer : Nat
  logits : FloatBinding alloc bytes initial views.logits logitsPointer
  labels : LabelBinding alloc bytes initial views.labels labelsPointer
  loss : TensorStorage alloc views.loss 4 lossPointer
  lse : TensorStorage alloc views.lse 4 lsePointer
  lossDtype : views.loss.dtype="torch.float32"
  lseDtype : views.lse.dtype="torch.float32"
  devices : views.logits.device=views.labels.device ∧
    views.logits.device=views.loss.device ∧ views.logits.device=views.lse.device
  labelsValid : ∀ row, row<p.rows → targetValue initial views.labels row=p.ignored ∨
    (0≤targetValue initial views.labels row ∧ targetValue initial views.labels row<(p.cols : Int))
  freshRegions : p.loss≠p.logits ∧ p.loss≠Region.cast p.labels ∧ p.lse≠Region.cast p.labels
  freshBacking : ∀ output∈[p.loss,p.lse], ∀ input∈[p.logits,Region.cast p.labels],
    (alloc output).Disjoint (alloc input)
  outputBacking : (alloc p.loss).Disjoint (alloc p.lse)

structure BackwardInputs (p : Parameters) (views : BackwardViews p)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte)
    (initial : State Word) where
  parameters : ParametersValid p
  scalars : RuntimeArgs.Values Word
  scalarBinding : RuntimeArgs.Bound p initial scalars
  byteCoherence : CoherentBytes alloc bytes
  typedViews : p.logits≠Region.cast p.labels ∧ p.lse≠Region.cast p.labels ∧ p.upstream≠Region.cast p.labels
  logitsPointer : Nat
  labelsPointer : Nat
  lsePointer : Nat
  upstreamPointer : Nat
  gradientPointer : Nat
  logits : FloatBinding alloc bytes initial views.logits logitsPointer
  labels : LabelBinding alloc bytes initial views.labels labelsPointer
  lse : FloatBinding alloc bytes initial views.lse lsePointer
  upstream : FloatBinding alloc bytes initial views.upstream upstreamPointer
  gradient : TensorStorage alloc views.gradient 4 gradientPointer
  gradientDtype : views.gradient.dtype="torch.float32"
  devices : views.logits.device=views.labels.device ∧ views.logits.device=views.lse.device ∧
    views.logits.device=views.upstream.device ∧ views.logits.device=views.gradient.device
  labelsValid : ∀ row, row<p.rows → targetValue initial views.labels row=p.ignored ∨
    (0≤targetValue initial views.labels row ∧ targetValue initial views.labels row<(p.cols : Int))
  savedRegions : p.gradient≠p.lse ∧ p.gradient≠p.upstream ∧ p.gradient≠Region.cast p.labels
  savedBacking : ∀ input∈[p.lse,p.upstream,Region.cast p.labels],
    (alloc p.gradient).Disjoint (alloc input)
  logitsAlias : (p.gradient=p.logits ∧ p.gradientBase=p.logitsBase ∧
    p.gradientStride=p.logitsStride) ∨
    (p.gradient≠p.logits ∧ (alloc p.gradient).Disjoint (alloc p.logits))

theorem matrix_numel (t : Tensor) (region : RegionName) (base rows cols stride row col : Nat)
    (view : MatrixView t region base rows cols stride) (hr : row<rows) (hc : col<cols) :
    row*cols+col<t.numel := by
  have bound := Nat.mul_le_mul_right cols (Nat.succ_le_of_lt hr)
  simp only [Nat.succ_mul] at bound
  simp only [Tensor.numel,view.shape,List.prod_cons,List.prod_nil,Nat.mul_one]
  omega

theorem logit_access_backed (p : Parameters) (views : ForwardViews p)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte)
    (initial : State Word) (inputs : ForwardInputs p views alloc bytes initial)
    (row col : Nat) (hr : row<p.rows) (hc : col<p.cols) :
    4*(Source.logitAddress p row col+1)≤(alloc p.logits).capacity := by
  have bound := inputs.logits.backed (row*p.cols+col)
    (matrix_numel _ _ _ _ _ _ _ _ views.logitsView hr hc)
  simpa only [matrix_index _ _ _ _ _ _ _ _ views.logitsView hr hc,
    views.logitsView.regionEq,Source.logitAddress] using bound

theorem target_bytes (p : Parameters) (views : ForwardViews p)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte)
    (initial : State Word) (inputs : ForwardInputs p views alloc bytes initial)
    (row : Nat) (hr : row<p.rows) :
    targetValue initial views.labels row=
      intAt (bytes (Region.cast p.labels)) (8*(p.labelsBase+row)) := by
  have hi : row<views.labels.numel := by
    simpa only [Tensor.numel,views.labelsView.shape,List.prod_cons,List.prod_nil,Nat.mul_one] using hr
  simpa only [vector_index _ _ _ _ _ _ views.labelsView hr,
    views.labelsView.regionEq,Nat.mul_one] using inputs.labels.cells row hi

/-- The input-only conversion needed by the historical scalar interpreter.
The FP32 compute source already uses the `.real` cell channel; dtype precision
is carried by `ComputeDType.fp32`, not by a mathematical interpretation. This definition is not a proof of
DSL execution equivalence, and public kernel acceptance must not treat it as one. -/
def scalarMemory {α : Type} [Inhabited α] (initial : State α) : Source.Memory α :=
  fun r i => ⟨fun d => (initial.mem r i).read d⟩

theorem logit_binding {α : Type} [Inhabited α] (p : Parameters) (initial : State α)
    (t : Tensor) (view : MatrixView t p.logits p.logitsBase p.rows p.cols p.logitsStride)
    (row col : Nat) (hr : row<p.rows) (hc : col<p.cols) :
    ((scalarMemory initial) p.logits (Source.logitAddress p row col)).read .real=
      fpValue initial t (row*p.cols+col) := by
  simp only [fpValue,matrix_index _ _ _ _ _ _ _ _ view hr hc,view.regionEq,
    scalarMemory,Source.Cell.read,Source.logitAddress]

theorem label_binding {α : Type} [Inhabited α] (p : Parameters) (initial : State α)
    (t : Tensor) (view : VectorView t (Region.cast p.labels) p.labelsBase p.rows 1)
    (row : Nat) (hr : row<p.rows) :
    Source.label p (scalarMemory initial) row=targetValue initial t row := by
  simp only [Source.label,targetValue,vector_index _ _ _ _ _ _ view hr,
    view.regionEq,Nat.mul_one,scalarMemory,Source.Cell.read]

theorem no_smoothing (p : Parameters) (valid : ParametersValid p)
    (disabled : p.hasSmoothing=Bool.false) : p.smoothing=0 := by
  have flag := valid.smoothingFlag
  rw [disabled] at flag
  have nonpos : ¬0<p.smoothing := by simpa using flag.symm
  exact le_antisymm (le_of_not_gt nonpos) valid.smoothingRange.1

#axiomsClean matrix_index
#axiomsClean vector_index
#axiomsClean output_index
#axiomsClean logit_binding
#axiomsClean label_binding
#axiomsClean no_smoothing
#axiomsClean logit_access_backed
#axiomsClean target_bytes
#specNonCircular TensorStorage avoiding [Source.forward,Source.backward,forwardSyntax,backwardSyntax]
#specNonCircular LabelBinding avoiding [Source.forward,Source.backward,forwardSyntax,backwardSyntax]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.KernelContract
