import bench.repairs.tritonbench_g.cross_entropy_ops.Kernels
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.KernelAccess

namespace VeriTile.Bench.Repairs.CrossEntropyOps.KernelContract
open VeriTile.Triton FP.Structural
open CrossEntropy1.TensorModel (Tensor)
noncomputable section
export CrossEntropy1.KernelContract (Word Byte Allocation fpValue targetValue wordAt intAt
  CoherentBytes TensorStorage FloatBinding LabelBinding MatrixView VectorView ScalarOutputView
  matrix_index vector_index output_index matrix_numel scalarMemory logit_binding label_binding)

abbrev Launch (p : Parameters) := CrossEntropy1.KernelContract.Launch p.toParameters
abbrev LaunchState {α : Type} (p : Parameters) (launch : Launch p) (initial : State α) :=
  CrossEntropy1.KernelContract.LaunchState p.toParameters launch initial
abbrev ParametersValid (p : Parameters) := CrossEntropy1.KernelContract.ParametersValid p.toParameters
abbrev BackwardViews (p : Parameters) := CrossEntropy1.KernelContract.BackwardViews p.toParameters

/-- The three actual forward argument views. The z-loss allocation exists in
both branches, but the split kernel performs no store to it. -/
structure ForwardViews (p : Parameters) extends CrossEntropy1.KernelContract.ForwardViews p.toParameters where
  zloss : Tensor
  zlossView : ScalarOutputView zloss p.zloss p.zlossBase p.rows (CrossEntropy1.Math.blocks p.cols p.block)
  zLse : p.zloss≠p.lse
  zLoss : p.zloss≠p.loss

/-- Ordinary successful preparation laws for kernel arguments. The inherited
contract binds actual FP32 bytes and signed int64 labels, permits original
storage offsets and aliases, and contains no execution/output premise. -/
structure ForwardInputs (p : Parameters) (views : ForwardViews p)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte) (words : State Word)
    extends CrossEntropy1.KernelContract.ForwardInputs p.toParameters views.toForwardViews alloc bytes words where
  logitScale : Word
  scaleBinding : words.regs .real [] "logit_scale"=some (fun _ => logitScale)
  zlossPointer : Nat
  zloss : TensorStorage alloc views.zloss 4 zlossPointer
  zlossDtype : views.zloss.dtype="torch.float32"
  zlossDevice : views.zloss.device=views.logits.device
  zFreshRegions : p.zloss≠p.logits ∧ p.zloss≠Region.cast p.labels
  zFreshBacking : ∀ input∈[p.logits,Region.cast p.labels,p.loss,p.lse], (alloc p.zloss).Disjoint (alloc input)

structure BackwardInputs (p : Parameters) (views : BackwardViews p)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte) (words : State Word)
    extends CrossEntropy1.KernelContract.BackwardInputs p.toParameters views alloc bytes words where
  logitScale : Word
  scaleBinding : words.regs .real [] "logit_scale"=some (fun _ => logitScale)
  savedLSELogits : p.lse≠p.logits
  savedLSEUpstream : p.lse≠p.upstream

abbrev forwardScalars {p : Parameters} {views : ForwardViews p} {alloc bytes words}
    (inputs : ForwardInputs p views alloc bytes words) : RuntimeArgs.Values Word :=
  ⟨inputs.scalars,inputs.logitScale⟩
abbrev backwardScalars {p : Parameters} {views : BackwardViews p} {alloc bytes words}
    (inputs : BackwardInputs p views alloc bytes words) : RuntimeArgs.Values Word :=
  ⟨inputs.scalars,inputs.logitScale⟩

theorem forward_binding {p : Parameters} {views : ForwardViews p} {alloc bytes words}
    (inputs : ForwardInputs p views alloc bytes words) :
    RuntimeArgs.Bound p words (forwardScalars inputs) :=
  ⟨inputs.scalarBinding,inputs.scaleBinding⟩
theorem backward_binding {p : Parameters} {views : BackwardViews p} {alloc bytes words}
    (inputs : BackwardInputs p views alloc bytes words) :
    RuntimeArgs.Bound p words (backwardScalars inputs) :=
  ⟨inputs.scalarBinding,inputs.scaleBinding⟩

theorem zloss_backed (p : Parameters) (views : ForwardViews p) (alloc bytes words)
    (inputs : ForwardInputs p views alloc bytes words) (launch : Launch p) :
    4*(p.zlossBase+launch.block*p.rows+launch.row+1)≤(alloc p.zloss).capacity :=
  CrossEntropy1.KernelAccess.output_backed _ _ _ _ _ _ _ _ _ views.zlossView inputs.zloss launch.rowBound launch.blockBound

#axiomsClean forward_binding
#axiomsClean backward_binding
#axiomsClean zloss_backed
#specNonCircular ForwardInputs avoiding [forwardSyntax,backwardSyntax,exec,run]
#specNonCircular BackwardInputs avoiding [forwardSyntax,backwardSyntax,exec,run]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.KernelContract
