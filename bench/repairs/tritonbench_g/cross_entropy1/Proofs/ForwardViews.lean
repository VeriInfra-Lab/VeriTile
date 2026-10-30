import bench.repairs.tritonbench_g.cross_entropy1.Proofs.BackwardBuffers

namespace VeriTile.Bench.Repairs.CrossEntropy1.Host
open VeriTile.Triton Source
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 6000000

/-- Local layout facts of ordinary materialization. A fresh resolved tensor
may preserve a transposed layout; the implication is used only when its last
stride is one. Contiguous outputs have nonoverlapping logical rows. -/
structure ViewGeometry {α σ : Type} (rt : Runtime α σ) : Prop where
  contiguousRows : ∀ input before output after rows cols, input.shape=[rows,cols] →
    rt.contiguous input before=.ok output after → rows≤1 ∨ cols≤rowStride output
  resolveRows : ∀ input before output after rows cols, input.shape=[rows,cols] →
    (rows≤1 ∨ cols≤rowStride input) → lastStride output=1 →
    rt.resolveNeg input before=.ok output after → rows≤1 ∨ cols≤rowStride output

theorem normalization_object {α σ : Type} (rt : Runtime α σ) (objects : ObjectRegistry rt)
    (input output : Tensor) (before after : World α σ) (valid : objects.valid input)
    (call : (if lastStride input≠1 then rt.contiguous input before else .ok input before)=Step.ok output after) :
    objects.valid output ∧ (output=input ∨ ¬before.live output.region) := by
  by_cases needed : lastStride input≠1
  · rw [if_pos needed] at call
    exact objects.contiguous _ _ _ _ call
  · rw [if_neg needed] at call
    cases call
    exact ⟨valid,Or.inl rfl⟩

theorem origin_separate {α σ : Type} (before : World α σ) (input output : Tensor) (r : RegionName)
    (origin : output=input ∨ ¬before.live output.region) (live : before.live r) (separate : input.region≠r) :
    output.region≠r := by
  rcases origin with same|fresh
  · simpa only [same] using separate
  · intro same
    exact fresh (same ▸ live)

structure ForwardViewData {α σ : Type} [Inhabited α] (obs : Observations α σ) {rt : Runtime α σ}
    (objects : ObjectRegistry rt) {M : FP32 α} {o : Options} {logits labels : Tensor}
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
    (lib : Library obs rt) (objects : ObjectRegistry rt) (views : ViewGeometry rt)
    {M : FP32 α} {o : Options} {logits labels : Tensor} {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs e)
    (logitsLive : initial.live logits.region) (rank : logits.strides.length=logits.shape.length)
    (registered : objects.valid logits) : ForwardViewData obs objects e := by
  have normalized := normalization_values obs rt lib logits e.inplaceLogits e.labelsReady e.normalized
    (k.labelsSnapshot.materialized.frame.live _ logitsLive) rank e.originalNormalize
  have original := normalized.1.prepend k.labelsSnapshot.materialized.frame logitsLive
  have resolved := lib.resolve _ _ _ _ original.live e.resolveCall
  have working := normalization_values obs rt lib e.resolvedLogits e.workingLogits e.resolvedState e.resolved
    resolved.1.live resolved.1.rank e.workingNormalize
  have firstObject := normalization_object rt objects logits e.inplaceLogits e.labelsReady e.normalized registered e.originalNormalize
  have resolvedObject := objects.resolve _ _ _ _ e.resolveCall
  have workingObject := normalization_object rt objects e.resolvedLogits e.workingLogits e.resolvedState e.resolved
    resolvedObject.1 e.workingNormalize
  obtain ⟨snapshot⟩ := snapshot_success rt labels e.labels e.ranked e.labelsReady e.snapshotCall
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
#axiomsClean normalization_object
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Host
