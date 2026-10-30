import bench.repairs.tritonbench_g.cross_entropy1.Proofs.ForwardValues
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.ForwardViews

namespace VeriTile.Bench.Repairs.CrossEntropy1.Host
open VeriTile.Triton Source
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 6000000

theorem local_lse_properties {α σ : Type} [Inhabited α] (obs : Observations α σ)
    (A : TensorArithmetic α) (rt : Runtime α σ) (ops : TensorLibrary obs A rt)
    (count rows : Nat) (losses lse : Tensor) (initial : World α σ) (pair : Tensor × Tensor) (final : World α σ)
    (sv : Nat → α) (saved : FloatArray obs initial lse (if count>1 then [count,rows] else [rows]) sv)
    (dense : lse.Dense) (trace : LocalTrace rt count losses lse initial pair final) :
    pair.2.Dense ∧ (pair.2=lse ∨ ¬initial.live pair.2.region) := by
  cases trace with
  | direct => exact ⟨dense,Or.inl rfl⟩
  | reduced many localLSE localLoss logged summed lseCall sumCall =>
    have shape : lse.shape=[count,rows] := by simpa only [if_pos many] using saved.shapeEq
    have first := ops.logsumexp lse count rows initial localLSE logged saved.live shape saved.dtype lseCall
    exact ⟨first.dense,Or.inr first.allocation.fresh⟩

theorem rank_lse_properties {α σ : Type} [Inhabited α] (obs : Observations α σ)
    (A : TensorArithmetic α) (rt : Runtime α σ) (lib : Library obs rt) (ops : TensorLibrary obs A rt)
    (gs gr : Nat → Nat) (comms : CollectiveLibrary obs A rt gs gr)
    (o : Options) (world rows : Nat) (localPair pair : Tensor × Tensor) (initial final : World α σ)
    (lv sv : Nat → α) (loss : FloatArray obs initial localPair.1 [rows] lv)
    (saved : FloatArray obs initial localPair.2 [rows] sv) (dense : localPair.2.Dense)
    (size : world>1 → world=gs (o.processGroup.getD 0))
    (trace : RankTrace rt o world rows localPair initial pair final) :
    pair.2.Dense ∧ (pair.2=localPair.2 ∨ ¬initial.live pair.2.region) := by
  cases trace with
  | direct => exact ⟨dense,Or.inl rfl⟩
  | reduced many gathered globalLSE handle allocated collected pending reduced waited allocationCall gatherCall sumCall lseCall waitCall =>
    have allocation := (lib.empty _ _ _ _ _ _ allocationCall).1
    have gather := comms.gather gathered localPair.2 (o.processGroup.getD 0) rows allocated collected
      allocation.live (allocation.frame.live _ saved.live) (by simpa [← size many] using allocation.shapeEq)
      saved.shapeEq (allocation.dtypeEq.trans saved.dtype) saved.dtype gatherCall
    have collectedFrame := allocation.frame.freshMutation gather.1 allocation.fresh
    have start := comms.start localPair.1 (o.processGroup.getD 0) rows collected handle pending
      (collectedFrame.live _ loss.live) loss.shapeEq loss.dtype sumCall
    have distinct := fresh_ne_live obs _ _ _ _ _ _ allocation _ loss.live
    have gatheredPending := gather.2.mutate start.1 distinct
    have produced := ops.logsumexp gathered world rows pending globalLSE reduced gatheredPending.live
      (by simpa [← size many] using gatheredPending.shapeEq) gatheredPending.dtype lseCall
    refine ⟨produced.dense,Or.inr ?_⟩
    intro live
    exact produced.allocation.fresh (start.1.live _ (collectedFrame.live _ live))

theorem aggregate_lse_properties {α σ : Type} [Inhabited α] (obs : Observations α σ)
    (A : TensorArithmetic α) (rt : Runtime α σ) (lib : Library obs rt) (ops : TensorLibrary obs A rt)
    (gs gr : Nat → Nat) (comms : CollectiveLibrary obs A rt gs gr)
    (o : Options) (world rows count : Nat) (labels losses lse : Tensor)
    (initial : World α σ) (returned : Tensor × Tensor) (final : World α σ) (lv sv : Nat → α)
    (loss : FloatArray obs initial losses (if count>1 then [count,rows] else [rows]) lv)
    (saved : FloatArray obs initial lse (if count>1 then [count,rows] else [rows]) sv)
    (dense : lse.Dense) (separate : losses.region≠lse.region)
    (size : world>1 → world=gs (o.processGroup.getD 0))
    (e : AggregateData rt o world rows count labels losses lse initial returned final) :
    returned.2.Dense ∧ (returned.2=lse ∨ ¬initial.live returned.2.region) := by
  have localValues := local_reduction_values obs A rt ops count rows losses lse initial e.localPair e.localState lv sv loss saved separate e.localCalls
  have localMeta := local_lse_properties obs A rt ops count rows losses lse initial e.localPair e.localState sv saved dense e.localCalls
  have ranks := rank_lse_properties obs A rt lib ops gs gr comms o world rows e.localPair e.rankPair e.localState e.reduced
    _ _ localValues.losses localValues.lse localMeta.1 size e.rankCalls
  refine ⟨by simpa only [e.lseIdentity] using ranks.1,?_⟩
  rw [e.lseIdentity]
  rcases ranks.2 with same|fresh
  · simpa only [same] using localMeta.2
  · right
    intro live
    exact fresh (localValues.frame.live _ live)

theorem forward_saved_frame {α σ : Type} [Inhabited α] (obs : Observations α σ)
    (A : TensorArithmetic α) (rt : Runtime α σ) (lib : Library obs rt) (ops : TensorLibrary obs A rt)
    (ints : IntegerViews obs) (gs gr : Nat → Nat) (comms : CollectiveLibrary obs A rt gs gr)
    {M : FP32 α} {o : Options} {logits labels : Tensor} {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs e)
    (labelDtype : labels.dtype="torch.int64") (size : e.world>1 → e.world=gs (o.processGroup.getD 0))
    (aggregation : ForwardAggregation e) (lossPeers lsePeers : Nat → α)
    (inputs : ForwardInputs comms aggregation lossPeers lsePeers) : Preserves obs e.resolved final := by
  have frame := forward_resolved_frame obs rt lib ints k
  have save := lib.save _ _ _ _ _ _ e.saveCall
  cases aggregation with
  | direct flag pair state => exact frame.trans (by simpa only [state] using save)
  | split flag trace =>
    have arrays := forward_arrays obs rt lib k
    have label := forward_label_array obs rt lib ints k labelDtype
    have aggregated := aggregation_values obs A rt lib ops gs gr comms o e.world e.rows e.count e.labels e.losses e.lse
      e.computed e.pair e.aggregated k.lossValues k.lseValues lossPeers lsePeers (obs.integer initial labels)
      arrays.1 arrays.2 label k.separation.outputDistinct.symm
      (by simpa only [ForwardData.parameters,forwardParameters,Region.cast_cast,Region.cast_id] using k.separation.lossLabels)
      size trace inputs
    have fresh := local_loss_fresh obs A rt ops e.count e.rows e.losses e.lse e.resolved e.computed trace.localPair trace.localState
      k.lossValues k.lseValues arrays.1 arrays.2 frame k.lossAllocation.fresh trace.localCalls
    exact (frame.freshMutation aggregated.2.1 fresh).trans save

theorem forward_saved_lse {α σ : Type} [Inhabited α] (obs : Observations α σ)
    (A : TensorArithmetic α) (rt : Runtime α σ) (lib : Library obs rt) (ops : TensorLibrary obs A rt)
    (ints : IntegerViews obs) (gs gr : Nat → Nat) (comms : CollectiveLibrary obs A rt gs gr)
    {M : FP32 α} {o : Options} {logits labels : Tensor} {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs e)
    (size : e.world>1 → e.world=gs (o.processGroup.getD 0)) (aggregation : ForwardAggregation e) :
    e.pair.2.Dense ∧ e.pair.2.region≠e.workingLogits.region := by
  have originalSeparate : e.lse.region≠e.workingLogits.region := k.separation.lseLogits
  cases aggregation with
  | direct flag pair state => simpa only [pair] using And.intro k.lseDense originalSeparate
  | split flag trace =>
    have arrays := forward_arrays obs rt lib k
    have properties := aggregate_lse_properties obs A rt lib ops gs gr comms o e.world e.rows e.count e.labels e.losses e.lse
      e.computed e.pair e.aggregated k.lossValues k.lseValues arrays.1 arrays.2 k.lseDense k.separation.outputDistinct.symm size trace
    refine ⟨properties.1,?_⟩
    rcases properties.2 with same|fresh
    · simpa only [same] using originalSeparate
    · intro same
      exact fresh (same.symm ▸ (forward_resolved_frame obs rt lib ints k).live _ k.working.live)

/-- The backward metadata preconditions are now derived from the exact saved
forward objects, including the original in-place destination and row guard. -/
theorem forward_backward_context {α σ : Type} [Inhabited α] (obs : Observations α σ)
    (A : TensorArithmetic α) (rt : Runtime α σ) (lib : Library obs rt) (ops : TensorLibrary obs A rt)
    (ints : IntegerViews obs) (objects : ObjectRegistry rt) (gs gr : Nat → Nat) (comms : CollectiveLibrary obs A rt gs gr)
    {M : FP32 α} {o : Options} {logits labels : Tensor} {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs e) (v : ForwardViewData obs objects e)
    (labelDtype : labels.dtype="torch.int64") (size : e.world>1 → e.world=gs (o.processGroup.getD 0))
    (aggregation : ForwardAggregation e) (lossPeers lsePeers : Nat → α)
    (inputs : ForwardInputs comms aggregation lossPeers lsePeers) :
    BackwardContext rt objects returned.context final := by
  have values := forward_values obs A rt lib ops ints gs gr comms k labelDtype size aggregation lossPeers lsePeers inputs
  have savedFrame := forward_saved_frame obs A rt lib ops ints gs gr comms k labelDtype size aggregation lossPeers lsePeers inputs
  have lse := forward_saved_lse obs A rt lib ops ints gs gr comms k size aggregation
  have contextEq : returned.context=⟨e.workingLogits,e.pair.2,e.labels,e.inplaceLogits,o,e.world*e.cols,e.rank*e.cols⟩ :=
    congrArg ForwardReturn.context e.returnIdentity
  have sourceLive := values.2.2.2.1.live
  have labelsLive := values.2.2.1.live
  have lseLive := values.2.1.live
  rw [contextEq] at sourceLive labelsLive lseLive ⊢
  refine ⟨sourceLive,labelsLive,lseLive,savedFrame.live _ (v.afterOriginal.live _ v.inplace.live),k.working.rank,k.notNeg,
    v.workingObject,v.inplaceObject,v.inplace.metadata.shape.trans k.working.metadata.shape.symm,v.workingLabels,lse.2.symm,?_⟩
  intro inplace rows cols shape
  have same : [rows,cols]=[e.rows,e.cols] := shape.symm.trans (k.working.metadata.shape.trans e.logitsShape)
  have eqs : rows=e.rows ∧ cols=e.cols := by simpa only [List.cons.injEq,and_true] using same
  simpa only [eqs.1,eqs.2] using v.workingRows inplace

#axiomsClean forward_saved_lse
#axiomsClean forward_saved_frame
#axiomsClean forward_backward_context
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Host
