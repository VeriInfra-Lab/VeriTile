import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.Preparation
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.ForwardViews

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Host
open VeriTile.Triton
open CrossEntropy1.Source (FP32)
open CrossEntropy1.Host (World Step Observations Materialized Preserves RowLayout ObjectRegistry
  ViewGeometry rowStride lastStride normalization_values normalization_object origin_separate snapshot_success)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

structure ForwardViewData {α σ : Type} [Inhabited α] (obs : Observations α σ) {rt : Runtime α σ}
    (objects : ObjectRegistry rt.toRuntime) {M : FP32 α} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    (e : ForwardData M rt o logits labels initial returned final) : Prop where
  inplace : Materialized obs logits e.inplaceLogits initial e.normalized
  afterOriginal : Preserves obs e.normalized e.resolved
  inplaceLayout : RowLayout e.inplaceLogits
  inplaceObject : objects.valid e.inplaceLogits
  workingObject : objects.valid e.workingLogits
  labelsObject : objects.valid e.labels
  inplaceLabels : e.inplaceLogits.region≠e.labels.region
  workingLabels : e.workingLogits.region≠e.labels.region
  workingRows : o.inplace=Bool.true → e.rows≤1 ∨ e.cols≤rowStride e.workingLogits

theorem forward_view_data {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (lib : LocalAPI obs rt) (objects : ObjectRegistry rt.toRuntime) (views : ViewGeometry rt.toRuntime)
    {M : FP32 α} {o : Options} {logits labels : Tensor} {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs e)
    (logitsLive : initial.live logits.region) (rank : logits.strides.length=logits.shape.length)
    (registered : objects.valid logits) : ForwardViewData obs objects e := by
  have normalized := normalization_values obs rt.toRuntime lib.tensor logits e.inplaceLogits e.labelsReady e.normalized
    (k.labelsSnapshot.materialized.frame.live _ logitsLive) rank e.originalNormalize
  have original := normalized.1.prepend k.labelsSnapshot.materialized.frame logitsLive
  have resolved := lib.tensor.resolve _ _ _ _ original.live e.resolveCall
  have working := normalization_values obs rt.toRuntime lib.tensor e.resolvedLogits e.workingLogits e.resolvedState e.resolved
    resolved.1.live resolved.1.rank e.workingNormalize
  have firstObject := normalization_object rt.toRuntime objects logits e.inplaceLogits e.labelsReady e.normalized registered e.originalNormalize
  have resolvedObject := objects.resolve _ _ _ _ e.resolveCall
  have workingObject := normalization_object rt.toRuntime objects e.resolvedLogits e.workingLogits e.resolvedState e.resolved
    resolvedObject.1 e.workingNormalize
  obtain ⟨snapshot⟩ := snapshot_success rt.toRuntime labels e.labels e.ranked e.labelsReady e.snapshotCall
  have labelObject := objects.clone _ _ _ _ snapshot.cloneCall
  have originalSeparate : logits.region≠e.labels.region := by
    intro same
    exact k.labelsSnapshot.fresh (same ▸ logitsLive)
  have inplaceSeparate := origin_separate e.labelsReady logits e.inplaceLogits e.labels.region firstObject.2
    k.labelsSnapshot.materialized.live originalSeparate
  have labelNormalized := normalized.1.frame.live _ k.labelsSnapshot.materialized.live
  have resolvedSeparate := origin_separate e.normalized e.inplaceLogits e.resolvedLogits e.labels.region resolvedObject.2
    labelNormalized inplaceSeparate
  have workingSeparate := origin_separate e.resolvedState e.resolvedLogits e.workingLogits e.labels.region workingObject.2
    (resolved.1.frame.live _ labelNormalized) resolvedSeparate
  refine ⟨original,resolved.1.frame.trans working.1.frame,normalized.2.1,firstObject.1,workingObject.1,labelObject,
    inplaceSeparate,workingSeparate,?_⟩
  intro inplace
  have guard := e.overlapGuard
  simp only [inplace,Bool.true_and,decide_eq_true_eq] at guard
  have originalRows : e.rows≤1 ∨ e.cols≤rowStride e.inplaceLogits := by omega
  have originalShape : e.inplaceLogits.shape=[e.rows,e.cols] := original.metadata.shape.trans e.logitsShape
  have resolvedShape : e.resolvedLogits.shape=[e.rows,e.cols] := resolved.1.metadata.shape.trans originalShape
  by_cases needed : lastStride e.resolvedLogits≠1
  · have call : rt.contiguous e.resolvedLogits e.resolvedState=.ok e.workingLogits e.resolved := by
      simpa only [if_pos needed] using e.workingNormalize
    exact views.contiguousRows _ _ _ _ _ _ resolvedShape call
  · have call := e.workingNormalize
    simp only [if_neg needed,Step.ok.injEq] at call
    have rows := views.resolveRows _ _ _ _ _ _ originalShape originalRows (not_not.mp needed) e.resolveCall
    simpa only [call.1.symm] using rows

#axiomsClean forward_view_data
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Host
