import bench.tritonbench_g.fast_ce_loss.Proofs.KernelABI
import bench.tritonbench_g.cross_entropy1.Proofs.KernelContract

namespace VeriTile.Bench.Repairs.FastCeLoss.KernelContract
open VeriTile.Triton FP.Structural KernelSource
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor coordinateOffset)
noncomputable section
set_option maxHeartbeats 4000000
namespace Storage
abbrev Allocation := VeriTile.Bench.Repairs.CrossEntropy1.KernelContract.Allocation
abbrev TensorStorage := VeriTile.Bench.Repairs.CrossEntropy1.KernelContract.TensorStorage
abbrev Byte := VeriTile.Bench.Repairs.CrossEntropy1.KernelContract.Byte
abbrev Word := VeriTile.Bench.Repairs.CrossEntropy1.KernelContract.Word
abbrev wordAt := VeriTile.Bench.Repairs.CrossEntropy1.KernelContract.wordAt
abbrev intAt := VeriTile.Bench.Repairs.CrossEntropy1.KernelContract.intAt
abbrev CoherentBytes := VeriTile.Bench.Repairs.CrossEntropy1.KernelContract.CoherentBytes
end Storage
open Storage

/-- Kernel regions start at the argument data pointer. Tensor indices remain
allocation-relative, so storage offsets are subtracted only at this boundary. -/
def tensorValue {α : Type} [Inhabited α] (s : State α) (t : Tensor) (i : Nat) : α :=
  (s.mem t.region (t.index i-t.storageOffset)).read .real

def tensorLabel {α : Type} [Inhabited α] (s : State α) (t : Tensor) (i : Nat) : Int :=
  (s.mem t.region (t.index i-t.storageOffset)).read .int

structure FloatArgument (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte)
    (s : State Word) (t : Tensor) (pointer : Nat) : Prop extends TensorStorage alloc t 4 pointer where
  dtype : t.dtype="torch.float32"
  cells : ∀ i, i<t.numel → tensorValue s t i=wordAt (bytes t.region) (4*t.index i)

structure LabelArgument (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte)
    (s : State Word) (t : Tensor) (pointer : Nat) : Prop extends TensorStorage alloc t 8 pointer where
  dtype : t.dtype="torch.int64"
  cells : ∀ i, i<t.numel → tensorLabel s t i=intAt (bytes t.region) (8*t.index i)

/-- Actual observed matrix strides. Only a used inner stride is required to
be one; singleton dimensions retain every stride admitted by Tensor metadata. -/
structure MatrixView (t : Tensor) (region : RegionName) (rows cols rowStride : Nat) where
  regionEq : t.region=region
  shape : t.shape=[rows,cols]
  innerStride : Nat
  strides : t.strides=[rowStride,innerStride]
  usedInner : 1<cols → innerStride=1

structure VectorView (t : Tensor) (region : RegionName) (count stride : Nat) : Prop where
  regionEq : t.region=region
  shape : t.shape=[count]
  strides : t.strides=[stride]

/-- Contiguous flat output/label indexing permits arbitrary unused strides. -/
structure FlatView (t : Tensor) (region : RegionName) (count : Nat) : Prop where
  regionEq : t.region=region
  numel : t.numel=count
  index : ∀ i, i<count → t.index i=t.storageOffset+i

 theorem matrix_index (t : Tensor) (region : RegionName) (rows cols stride row col : Nat)
    (view : MatrixView t region rows cols stride) (hr : row<rows) (hc : col<cols) :
    t.index (row*cols+col)=t.storageOffset+row*stride+col := by
  have quotient : (row*cols+col)/cols=row := by
    apply Nat.div_eq_of_lt_le
    · omega
    · rw [Nat.add_mul,Nat.one_mul]
      omega
  have remainder : (row*cols+col)%cols=col := by
    have h := Nat.div_add_mod (row*cols+col) cols
    rw [quotient,Nat.mul_comm cols row] at h
    omega
  have inner : col*view.innerStride=col := by
    by_cases many : 1<cols
    · simp [view.usedInner many]
    · have zero : col=0 := by omega
      simp [zero]
  simp [Tensor.index,coordinateOffset,view.shape,view.strides,
    quotient,remainder,Nat.mod_eq_of_lt hr,Nat.mod_eq_of_lt hc,inner,Nat.add_assoc]

 theorem matrix_numel (t : Tensor) (region : RegionName) (rows cols stride row col : Nat)
    (view : MatrixView t region rows cols stride) (hr : row<rows) (hc : col<cols) :
    row*cols+col<t.numel := by
  have bound := Nat.mul_le_mul_right cols (Nat.succ_le_of_lt hr)
  simp only [Nat.succ_mul] at bound
  simp only [Tensor.numel,view.shape,List.prod_cons,List.prod_nil,Nat.mul_one]
  omega

 theorem vector_index (t : Tensor) (region : RegionName) (count stride i : Nat)
    (view : VectorView t region count stride) (hi : i<count) :
    t.index i=t.storageOffset+i*stride := by
  simp [Tensor.index,coordinateOffset,view.shape,view.strides,Nat.mod_eq_of_lt hi]

 theorem singleton_matrix_row (t : Tensor) (region : RegionName) (cols stride col : Nat)
    (view : MatrixView t region 1 cols stride) (hc : col<cols) :
    t.index col=t.storageOffset+col := by
  simpa using matrix_index t region 1 cols stride 0 col view (by omega) hc

 theorem singleton_vector (t : Tensor) (region : RegionName) (stride : Nat)
    (view : VectorView t region 1 stride) : t.index 0=t.storageOffset := by
  simpa using vector_index t region 1 stride 0 view (by omega)

structure ParametersValid (p : Parameters) : Prop where
  vocabPositive : 0<p.vocab
  rowsLimit : p.rows≤2^31-1
  vocabLimit : p.vocab≤4096*65535
  blockPositive : 0<p.block
  blockLimit : p.block≤65536
  rowStrideNative : p.logitsStride<2^63
  upstreamStrideNative : p.upstreamStride<2^63

structure Launch (p : Parameters) (blocks : Nat) where
  row : Nat
  block : Nat
  rowBound : row<p.rows
  blockBound : block<blocks

def LaunchState {α : Type} (launch : Launch p blocks) (s : State α) : Prop :=
  s.pids 0=launch.row ∧ s.pids 1=launch.block

structure ForwardViews (p : Parameters) (chunks : Nat) where
  logits : Tensor
  labels : Tensor
  loss : Tensor
  lse : Tensor
  logitsView : MatrixView logits p.logits p.rows p.vocab p.logitsStride
  labelsView : FlatView labels (Region.cast p.labels) p.rows
  lossView : FlatView loss p.loss p.rows
  lseView : FlatView lse p.lse (p.rows*chunks)
  outputsDistinct : p.loss≠p.lse
  lossLogits : p.loss≠p.logits
  lseLogits : p.lse≠p.logits

structure BackwardViews (p : Parameters) where
  logits : Tensor
  labels : Tensor
  lse : Tensor
  upstream : Tensor
  logitsView : MatrixView logits p.logits p.rows p.vocab p.logitsStride
  labelsView : FlatView labels (Region.cast p.labels) p.rows
  lseView : FlatView lse p.lse p.rows
  upstreamView : VectorView upstream p.upstream p.rows p.upstreamStride

structure ForwardInputs (p : Parameters) (views : ForwardViews p chunks)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte) (words : State Word) where
  parameters : ParametersValid p
  options : KernelABI.Options p
  byteCoherence : CoherentBytes alloc bytes
  logitsPointer : Nat
  labelsPointer : Nat
  lossPointer : Nat
  lsePointer : Nat
  logits : FloatArgument alloc bytes words views.logits logitsPointer
  labels : LabelArgument alloc bytes words views.labels labelsPointer
  loss : TensorStorage alloc views.loss 4 lossPointer
  lse : TensorStorage alloc views.lse 4 lsePointer
  lossDtype : views.loss.dtype="torch.float32"
  lseDtype : views.lse.dtype="torch.float32"
  devices : views.logits.device=views.labels.device ∧ views.logits.device=views.loss.device ∧
    views.logits.device=views.lse.device
  labelsValid : ∀ row, row<p.rows → tensorLabel words views.labels row = -100 ∨
    (0≤tensorLabel words views.labels row ∧ tensorLabel words views.labels row<(p.vocab : Int))
  labelSeparation : p.logits≠Region.cast p.labels ∧ p.loss≠Region.cast p.labels ∧ p.lse≠Region.cast p.labels
  distinctBacking : ∀ r∈[p.logits,Region.cast p.labels,p.loss,p.lse],
    ∀ s∈[p.logits,Region.cast p.labels,p.loss,p.lse], r≠s → (alloc r).Disjoint (alloc s)

/-- These are prepared kernel arguments. The row upstream may be a stride-zero
expansion. Its values and saved LSE are supplied inputs, without host formulas. -/
structure BackwardInputs (p : Parameters) (views : BackwardViews p)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte) (words : State Word) where
  parameters : ParametersValid p
  options : KernelABI.Options p
  byteCoherence : CoherentBytes alloc bytes
  logitsPointer : Nat
  labelsPointer : Nat
  lsePointer : Nat
  upstreamPointer : Nat
  logits : FloatArgument alloc bytes words views.logits logitsPointer
  labels : LabelArgument alloc bytes words views.labels labelsPointer
  lse : FloatArgument alloc bytes words views.lse lsePointer
  upstream : FloatArgument alloc bytes words views.upstream upstreamPointer
  devices : views.logits.device=views.labels.device ∧ views.logits.device=views.lse.device ∧
    views.logits.device=views.upstream.device
  labelsValid : ∀ row, row<p.rows → tensorLabel words views.labels row = -100 ∨
    (0≤tensorLabel words views.labels row ∧ tensorLabel words views.labels row<(p.vocab : Int))
  typedSeparation : p.logits≠Region.cast p.labels ∧ p.lse≠Region.cast p.labels ∧
    p.upstream≠Region.cast p.labels
  savedSeparation : p.logits≠p.lse ∧ p.logits≠p.upstream
  savedBacking : (alloc p.logits).Disjoint (alloc p.lse) ∧
    (alloc p.logits).Disjoint (alloc p.upstream) ∧
    (alloc p.logits).Disjoint (alloc (Region.cast p.labels))

#axiomsClean matrix_index
#axiomsClean vector_index
#axiomsClean singleton_matrix_row
#axiomsClean singleton_vector
#specNonCircular FloatArgument avoiding [forward,chunked,backward,FP.Structural.exec,FP.Structural.run]
#specNonCircular LabelArgument avoiding [forward,chunked,backward,FP.Structural.exec,FP.Structural.run]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.KernelContract
