import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.AggregationValues
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.ForwardArrays
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.SavedUnsplit
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.SavedMetadata

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Host
open CrossEntropy1.Source (FP32)
open CrossEntropy1.Host (World Step Observations TensorArithmetic TensorLibrary CollectiveLibrary IntegerViews
  ObjectRegistry FloatArray Preserves RankInputs localResult rankResult local_loss_fresh
  local_reduction_values local_lse_properties rank_lse_properties BackwardContext)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

theorem forward_split_trace {α σ : Type} [Inhabited α] {M : FP32 α} {rt : Runtime α σ} {o : Options}
    {logits labels : Tensor} {initial final : World α σ} {returned : ForwardReturn}
    (e : ForwardData M rt o logits labels initial returned final) (split : e.world>1 ∨ e.cols>65536) :
    Nonempty (AggregateData rt o e.world e.rows e.count e.labels e.losses e.lse e.computed e.result e.aggregated) := by
  have call := e.aggregateCall
  simp only [split,decide_true,↓reduceIte] at call
  exact aggregate_success rt o e.world e.rows e.count e.labels e.losses e.lse e.computed e.result e.aggregated call

/-- The actual split/distributed returned arrays, plus a frame for every
allocation existing before output allocation. Collective inputs here are
transmitted words; the Real theorem must bind them to actual peer executions. -/
theorem returned_split_values {α σ : Type} [Inhabited α] (obs : Observations α σ)
    (A : TensorArithmetic α) (rt : Runtime α σ) (lib : LocalAPI obs rt)
    (ops : TensorLibrary obs A rt.toRuntime) (zeros : ZeroAPI obs rt A.zero) (ints : IntegerViews obs)
    (gs gr : Nat → Nat) (comms : CollectiveLibrary obs A rt.toRuntime gs gr)
    {M : FP32 α} {o : Options} {logits labels : Tensor} {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs e)
    (labelDtype : labels.dtype="torch.int64") (size : e.world>1 → e.world=gs (o.processGroup.getD 0))
    (a : AggregateData rt o e.world e.rows e.count e.labels e.losses e.lse e.computed e.result e.aggregated)
    (lossPeers lsePeers : Nat → α) (inputs : RankInputs comms a.rankCalls lossPeers lsePeers) :
    FloatArray obs final returned.losses [e.rows]
      (aggregateLossValue A o e.world e.count e.rows k.lossValues k.lseValues lossPeers lsePeers (obs.integer initial labels)) ∧
    FloatArray obs final returned.context.lse [e.rows]
      (rankResult A.logsumexp e.world e.rows (localResult A.logsumexp e.count e.rows k.lseValues) lsePeers) ∧
    FloatArray obs final returned.zLosses [e.rows]
      (fun row => correctionZ A o (obs.integer initial labels row)
        (rankResult A.logsumexp e.world e.rows (localResult A.logsumexp e.count e.rows k.lseValues) lsePeers row)) ∧
    CrossEntropy1.Host.IntArray obs final returned.context.labels [e.rows] (obs.integer initial labels) ∧
    Preserves obs e.resolved final := by
  have arrays := forward_arrays obs rt lib k
  have labelArray := forward_label_array obs rt lib ints k labelDtype
  have values := aggregation_values obs A rt lib ops zeros gs gr comms o e.world e.rows e.count e.labels e.losses e.lse
    e.computed e.result e.aggregated k.lossValues k.lseValues lossPeers lsePeers (obs.integer initial labels)
    arrays.1 arrays.2 labelArray k.separation.outputDistinct.symm
    (by simpa only [ForwardData.parameters,forwardParameters,CrossEntropy1.Host.forwardParameters,
      VeriTile.Triton.Region.cast_cast,VeriTile.Triton.Region.cast_id] using k.separation.lossLabels) size a inputs
  have post := (lib.tensor.save _ _ _ _ _ _ e.saveCall).trans (lib.mark _ _ _ e.markCall)
  have frame := forward_resolved_frame obs rt lib ints k
  have fresh := local_loss_fresh obs A rt.toRuntime ops e.count e.rows e.losses e.lse e.resolved e.computed
    a.localPair a.localState k.lossValues k.lseValues arrays.1 arrays.2 frame k.lossAllocation.fresh a.localCalls
  refine ⟨?_,?_,?_,?_,(frame.freshMutation values.2.1 fresh).trans post⟩
  · simpa only [e.returnIdentity] using values.2.2.1.preserve post
  · simpa only [e.returnIdentity] using values.2.2.2.1.preserve post
  · simpa only [e.returnIdentity] using values.2.2.2.2.1.preserve post
  · simpa only [e.returnIdentity] using values.2.2.2.2.2.1.preserve post

theorem split_saved {α σ : Type} [Inhabited α] (obs : Observations α σ)
    (A : TensorArithmetic α) (rt : Runtime α σ) (lib : LocalAPI obs rt)
    (ops : TensorLibrary obs A rt.toRuntime) (zeros : ZeroAPI obs rt A.zero) (ints : IntegerViews obs)
    (objects : ObjectRegistry rt.toRuntime) (gs gr : Nat → Nat) (comms : CollectiveLibrary obs A rt.toRuntime gs gr)
    {M : FP32 α} {o : Options} {logits labels : Tensor} {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs e)
    (v : ForwardViewData obs objects e) (labelDtype : labels.dtype="torch.int64")
    (size : e.world>1 → e.world=gs (o.processGroup.getD 0))
    (a : AggregateData rt o e.world e.rows e.count e.labels e.losses e.lse e.computed e.result e.aggregated)
    (lossPeers lsePeers : Nat → α) (inputs : RankInputs comms a.rankCalls lossPeers lsePeers) :
    SavedForwardData obs rt objects logits labels initial final returned e.rows := by
  have values := returned_split_values obs A rt lib ops zeros ints gs gr comms k labelDtype size a lossPeers lsePeers inputs
  have arrays := forward_arrays obs rt lib k
  have locals := local_reduction_values obs A rt.toRuntime ops e.count e.rows e.losses e.lse e.computed a.localPair a.localState
    k.lossValues k.lseValues arrays.1 arrays.2 k.separation.outputDistinct.symm a.localCalls
  have localMeta := local_lse_properties obs A rt.toRuntime ops e.count e.rows e.losses e.lse e.computed a.localPair a.localState
    k.lseValues arrays.2 k.lseDense a.localCalls
  have ranks := rank_lse_properties obs A rt.toRuntime lib.tensor ops gs gr comms o.toOptions e.world e.rows
    a.localPair a.rankPair a.localState a.reduced _ _ locals.losses locals.lse localMeta.1 size a.rankCalls
  have origin : a.rankPair.2=e.lse ∨ ¬e.computed.live a.rankPair.2.region := by
    rcases ranks.2 with same|fresh
    · simpa only [same] using localMeta.2
    · exact Or.inr (fun live => fresh (locals.frame.live _ live))
  have lseSeparate : e.result.lse.region≠e.workingLogits.region := by
    rw [a.lseIdentity]
    rcases origin with same|fresh
    · simpa only [same] using k.separation.lseLogits
    · intro same
      exact fresh (same.symm ▸ (forward_resolved_frame obs rt lib ints k).live _ k.working.live)
  have working := materialized_preserve k.working values.2.2.2.2
  have contextEq : returned.context=⟨e.workingLogits,e.result.lse,e.labels,e.inplaceLogits,o,e.world*e.cols,e.rank*e.cols⟩ :=
    congrArg ForwardReturn.context e.returnIdentity
  have ctx : BackwardContext rt.toRuntime objects returned.context.base final := by
    have sourceLive := working.live
    have labelsLive := values.2.2.2.1.live
    have lseLive := values.2.1.live
    simp only [contextEq] at labelsLive lseLive
    simp only [contextEq,Context.base]
    refine ⟨sourceLive,labelsLive,lseLive,values.2.2.2.2.live _ (v.afterOriginal.live _ v.inplace.live),
      k.working.rank,k.notNeg,v.workingObject,v.inplaceObject,
      v.inplace.metadata.shape.trans k.working.metadata.shape.symm,v.workingLabels,lseSeparate.symm,?_⟩
    intro inplace rows cols shape
    have same : [rows,cols]=[e.rows,e.cols] := shape.symm.trans (k.working.metadata.shape.trans e.logitsShape)
    have eqs : rows=e.rows ∧ cols=e.cols := by simpa only [List.cons.injEq,and_true] using same
    simpa only [eqs.1,eqs.2] using v.workingRows inplace
  refine ⟨ctx,?_,?_,values.2.2.2.1,?_,values.2.1.shapeEq,values.2.1.dtype,?_⟩
  · simpa only [contextEq] using working
  · simpa only [contextEq] using k.layout
  · simpa only [contextEq] using k.labelsSnapshot.dense
  · simpa only [contextEq,a.lseIdentity] using ranks.1

#axiomsClean forward_split_trace
#axiomsClean returned_split_values
#axiomsClean split_saved
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Host
