import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.Correction
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.AggregationValues

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Host
open CrossEntropy1.Host (World Observations TensorArithmetic TensorLibrary CollectiveLibrary
  FloatArray IntArray Mutation RankInputs localResult rankResult local_reduction_values
  local_loss_separate rank_reduction_values mask_values)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

def aggregateLossValue {α : Type} (A : TensorArithmetic α) (o : Options)
    (world count rows : Nat) (lv sv lossPeers lsePeers : Nat → α) (targets : Nat → Int) (row : Nat) : α :=
  let loss := rankResult A.rankSum world rows (localResult A.sum count rows lv) lossPeers row
  let lse := rankResult A.logsumexp world rows (localResult A.logsumexp count rows sv) lsePeers row
  if targets row=o.ignored then A.zero else correctionLoss A o (targets row) (A.add loss lse) lse

/-- Composition follows the actual local reductions, collectives, wait,
in-place addition, correction and mask. All numerical premises are individual
operation contracts or transmitted peer inputs; no final task value is assumed. -/
theorem aggregation_values {α σ : Type} [Inhabited α] (obs : Observations α σ)
    (A : TensorArithmetic α) (rt : Runtime α σ) (lib : LocalAPI obs rt) (ops : TensorLibrary obs A rt.toRuntime) (zeros : ZeroAPI obs rt A.zero)
    (gs gr : Nat → Nat) (comms : CollectiveLibrary obs A rt.toRuntime gs gr)
    (o : Options) (world rows count : Nat) (labels losses lse : Tensor)
    (initial : World α σ) (returned : Aggregated) (final : World α σ)
    (lv sv lossPeers lsePeers : Nat → α) (targets : Nat → Int)
    (loss : FloatArray obs initial losses (if count>1 then [count,rows] else [rows]) lv)
    (saved : FloatArray obs initial lse (if count>1 then [count,rows] else [rows]) sv)
    (label : IntArray obs initial labels [rows] targets)
    (separate : losses.region≠lse.region) (labelsSeparate : losses.region≠labels.region)
    (size : world>1 → world=gs (o.processGroup.getD 0))
    (e : AggregateData rt o world rows count labels losses lse initial returned final)
    (inputs : RankInputs comms e.rankCalls lossPeers lsePeers) :
    returned.losses=e.localPair.1 ∧ Mutation obs initial final e.localPair.1 ∧
      FloatArray obs final returned.losses [rows]
        (aggregateLossValue A o world count rows lv sv lossPeers lsePeers targets) ∧
      FloatArray obs final returned.lse [rows]
        (rankResult A.logsumexp world rows (localResult A.logsumexp count rows sv) lsePeers) ∧
      FloatArray obs final returned.zLosses [rows]
        (fun row => correctionZ A o (targets row)
          (rankResult A.logsumexp world rows (localResult A.logsumexp count rows sv) lsePeers row)) ∧
      IntArray obs final labels [rows] targets ∧ returned.losses.region≠returned.lse.region := by
  have localValues := local_reduction_values obs A rt.toRuntime ops count rows losses lse initial e.localPair e.localState
    lv sv loss saved separate e.localCalls
  have localSeparate := local_loss_separate obs A rt.toRuntime ops count rows losses lse initial e.localPair e.localState
    lv sv loss saved labels.region label.live labelsSeparate e.localCalls
  have ranked := rank_reduction_values obs A rt.toRuntime lib.tensor ops gs gr comms o.toOptions world rows e.localPair e.rankPair
    e.localState e.reduced _ _ lossPeers lsePeers localValues.losses localValues.lse localValues.separate size e.rankCalls inputs
  have labelRanked := (label.preserve localValues.frame).mutate ranked.2.1 localSeparate.symm
  have addition := ops.add e.rankPair.1 e.rankPair.2 e.reduced e.addedLoss e.added ranked.2.2.1.live ranked.2.2.2.1.live
    ranked.2.2.1.dtype ranked.2.2.2.1.dtype (ranked.2.2.1.shapeEq.trans ranked.2.2.2.1.shapeEq.symm) e.addCall
  let rankLoss := rankResult A.rankSum world rows (localResult A.sum count rows lv) lossPeers
  let rankLSE := rankResult A.logsumexp world rows (localResult A.logsumexp count rows sv) lsePeers
  have added : FloatArray obs e.added e.addedLoss [rows] (fun row => A.add (rankLoss row) (rankLSE row)) := by
    rw [addition.1]
    refine ⟨ranked.2.2.1.shapeEq,ranked.2.2.1.dtype,addition.2.1.live _ ranked.2.2.1.live,?_⟩
    intro row hr
    have lseBound : row<e.rankPair.2.numel := by
      simpa only [Tensor.numel,ranked.2.2.1.shapeEq,ranked.2.2.2.1.shapeEq] using hr
    rw [addition.2.2 row hr,ranked.2.2.1.values row hr,ranked.2.2.2.1.values row lseBound]
  have lseAdded := ranked.2.2.2.1.mutate addition.2.1 ranked.2.2.2.2.symm
  have labelRankSeparate : labels.region≠e.rankPair.1.region := by simpa only [ranked.1] using localSeparate.symm
  have labelAdded := labelRanked.mutate addition.2.1 labelRankSeparate
  have corrected := correction_values obs A rt ops zeros o rows labels e.addedLoss e.rankPair.2 e.added e.correctedState e.corrected
    _ rankLSE targets added lseAdded labelAdded (by simpa only [addition.1] using ranked.2.2.2.2.symm) e.correctionCalls
  have labelCorrected := labelAdded.mutate corrected.2.1 (by simpa only [addition.1] using labelRankSeparate)
  have masked := mask_values obs A rt.toRuntime ops rows o.ignored e.corrected.1 labels e.mask returned.losses
    e.correctedState e.compared final _ targets corrected.2.2.1 labelCorrected e.compareCall e.maskCall
  have outputIdentity : returned.losses=e.localPair.1 := masked.1.trans (corrected.1.trans (addition.1.trans ranked.1))
  have correctionDst : e.addedLoss=e.localPair.1 := addition.1.trans ranked.1
  have maskDst : e.corrected.1=e.localPair.1 := corrected.1.trans correctionDst
  have allMutation : Mutation obs initial final e.localPair.1 := by
    have a : Mutation obs e.reduced e.added e.localPair.1 := by simpa only [ranked.1] using addition.2.1
    have c : Mutation obs e.added e.correctedState e.localPair.1 := by simpa only [correctionDst] using corrected.2.1
    have m : Mutation obs e.correctedState final e.localPair.1 := by simpa only [maskDst] using masked.2.1
    exact (((ranked.2.1.prepend localValues.frame).trans a).trans c).trans m
  have lseFinal := corrected.2.2.2.2.1.mutate masked.2.1
    (by simpa only [corrected.1,addition.1] using ranked.2.2.2.2.symm)
  have labelFinal := labelCorrected.mutate masked.2.1
    (by simpa only [maskDst] using localSeparate.symm)
  have zFinal := corrected.2.2.2.1.mutate masked.2.1
    (by simpa only [corrected.1] using corrected.2.2.2.2.2)
  refine ⟨outputIdentity,allMutation,?_,?_,?_,labelFinal,?_⟩
  · exact masked.2.2
  · simpa only [e.lseIdentity] using lseFinal
  · simpa only [e.zIdentity] using zFinal
  · simpa only [masked.1,corrected.1,addition.1,e.lseIdentity] using ranked.2.2.2.2

#axiomsClean aggregation_values
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Host
