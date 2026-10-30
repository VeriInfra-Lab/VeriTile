import bench.tritonbench_g.cross_entropy2.Kernels
import bench.tritonbench_g.cross_entropy1.Proofs.KernelContract

namespace VeriTile.Bench.Repairs.CrossEntropy2.KernelContract
open VeriTile.Triton FP.Structural
open CrossEntropy1.TensorModel (Tensor)
noncomputable section
export CrossEntropy1.KernelContract (Word Byte Allocation CoherentBytes TensorStorage FloatBinding
  MatrixView VectorView ScalarOutputView fpValue targetValue wordAt matrix_index vector_index output_index
  matrix_numel scalarMemory logit_binding label_binding)

/-- The source int32 label load and explicit signed int64 widening observe
these exact four backing bytes, including a label view aliased with FP32 input. -/
structure LabelBinding (alloc : RegionName → Allocation)
    (bytes : RegionName → Nat → Byte) (initial : State Word) (t : Tensor)
    (pointer : Nat) : Prop extends TensorStorage alloc t 4 pointer where
  dtype : t.dtype="torch.int32"
  cells : ∀ i, i<t.numel → targetValue initial t i=(wordAt (bytes t.region) (4*t.index i)).toInt

structure ParametersValid (p : Parameters) : Prop where
  geometry : CrossEntropy1.Native.Supported p.rows (CrossEntropy1.Math.blocks p.cols p.block) p.block p.cols
  powerOfTwo : ∃ k, p.block=2^k
  classesPositive : 0<p.cols
  totalClasses : 0<p.totalClasses ∧ p.totalClasses<2^63
  classInterval : 0≤p.classStart ∧ p.classStart+(p.cols : Int)≤(p.totalClasses : Int)
  smoothingRange : 0≤p.smoothing ∧ p.smoothing≤1
  scaleNonnegative : 0≤p.squareScale
  ignoredRange : -(2^63 : Int)≤p.ignored ∧ p.ignored<2^63
  stridesRepresentable : p.logitsStride<2^63 ∧ p.gradientStride<2^63 ∧ p.upstreamStride<2^63

structure Launch (p : Parameters) where
  row : Nat
  block : Nat
  rowBound : row<p.rows
  blockBound : block<CrossEntropy1.Math.blocks p.cols p.block

def LaunchState {α : Type} (p : Parameters) (launch : Launch p) (initial : State α) : Prop :=
  initial.pids 0=launch.row ∧ initial.pids 1=launch.block

structure ForwardViews (p : Parameters) extends CrossEntropy1.KernelContract.ForwardViews p.toParameters where
  zLoss : Tensor
  zLossView : ScalarOutputView zLoss p.zLoss p.zLossBase p.rows (CrossEntropy1.Math.blocks p.cols p.block)
  zDistinct : p.lse≠p.zLoss ∧ p.loss≠p.zLoss

abbrev BackwardViews (p : Parameters) := CrossEntropy1.KernelContract.BackwardViews p.toParameters

/-- Actual supplied argument views, their physical pointers and initial bytes.
No host operation, result, execution or numerical equality is a field. -/
structure ForwardInputs (p : Parameters) (views : ForwardViews p)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte) (initial : State Word) where
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
  zLossPointer : Nat
  logits : FloatBinding alloc bytes initial views.logits logitsPointer
  labels : LabelBinding alloc bytes initial views.labels labelsPointer
  loss : TensorStorage alloc views.loss 4 lossPointer
  lse : TensorStorage alloc views.lse 4 lsePointer
  zLoss : TensorStorage alloc views.zLoss 4 zLossPointer
  lossDtype : views.loss.dtype="torch.float32"
  lseDtype : views.lse.dtype="torch.float32"
  zLossDtype : views.zLoss.dtype="torch.float32"
  devices : views.logits.device=views.labels.device ∧ views.logits.device=views.loss.device ∧
    views.logits.device=views.lse.device ∧ views.logits.device=views.zLoss.device
  labelsValid : ∀ row, row<p.rows → targetValue initial views.labels row=p.ignored ∨
    (0≤targetValue initial views.labels row ∧ targetValue initial views.labels row<(p.totalClasses : Int))
  freshRegions : p.loss≠p.logits ∧ p.loss≠Region.cast p.labels ∧ p.lse≠Region.cast p.labels ∧
    p.zLoss≠p.logits ∧ p.zLoss≠Region.cast p.labels
  freshBacking : ∀ output∈[p.loss,p.lse,p.zLoss], ∀ input∈[p.logits,Region.cast p.labels],
    (alloc output).Disjoint (alloc input)
  outputBacking : (alloc p.loss).Disjoint (alloc p.lse) ∧
    (alloc p.loss).Disjoint (alloc p.zLoss) ∧ (alloc p.lse).Disjoint (alloc p.zLoss)

structure BackwardInputs (p : Parameters) (views : BackwardViews p)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte) (initial : State Word) where
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
    (0≤targetValue initial views.labels row ∧ targetValue initial views.labels row<(p.totalClasses : Int))
  freshRegions : p.gradient≠p.logits ∧ p.gradient≠p.lse ∧ p.gradient≠p.upstream ∧ p.gradient≠Region.cast p.labels
  freshBacking : ∀ input∈[p.logits,p.lse,p.upstream,Region.cast p.labels],
    (alloc p.gradient).Disjoint (alloc input)

#specNonCircular ForwardInputs avoiding [Source.forward,Source.backward,forwardSyntax,backwardSyntax]
#specNonCircular BackwardInputs avoiding [Source.forward,Source.backward,forwardSyntax,backwardSyntax]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.KernelContract
