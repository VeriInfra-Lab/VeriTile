import bench.tritonbench_g.cross_entropy1.Proofs.Collectives
import bench.tritonbench_g.cross_entropy1.Proofs.Mask

namespace VeriTile.Bench.Repairs.CrossEntropy1.Host
open VeriTile.Triton Source
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 6000000

theorem local_loss_separate {α σ : Type} [Inhabited α] (obs : Observations α σ)
    (A : TensorArithmetic α) (rt : Runtime α σ) (ops : TensorLibrary obs A rt)
    (count rows : Nat) (losses lse : Tensor) (initial : World α σ) (pair : Tensor × Tensor) (final : World α σ)
    (lv sv : Nat → α)
    (loss : FloatArray obs initial losses (if count>1 then [count,rows] else [rows]) lv)
    (saved : FloatArray obs initial lse (if count>1 then [count,rows] else [rows]) sv)
    (r : RegionName) (live : initial.live r) (different : losses.region≠r)
    (trace : LocalTrace rt count losses lse initial pair final) : pair.1.region≠r := by
  cases trace with
  | direct => exact different
  | reduced many localLSE localLoss logged summed lseCall sumCall =>
    have si : lse.shape=[count,rows] := by simpa only [if_pos many] using saved.shapeEq
    have li : losses.shape=[count,rows] := by simpa only [if_pos many] using loss.shapeEq
    have first := ops.logsumexp lse count rows initial localLSE logged saved.live si saved.dtype lseCall
    have second := ops.sum losses count rows logged localLoss _ (first.allocation.frame.live _ loss.live) li loss.dtype sumCall
    exact fresh_ne_live obs _ _ _ _ _ _ second.allocation _ (first.allocation.frame.live _ live)

def aggregateLossValue {α : Type} (A : TensorArithmetic α) (o : Options)
    (world count rows : Nat) (lv sv lossPeers lsePeers : Nat → α) (targets : Nat → Int) (row : Nat) : α :=
  let loss := rankResult A.rankSum world rows (localResult A.sum count rows lv) lossPeers row
  let lse := rankResult A.logsumexp world rows (localResult A.logsumexp count rows sv) lsePeers row
  if targets row=o.ignored then A.zero else correctedValue A o.squareScale (A.add loss lse) lse

/-- Composition follows the actual local reductions, collectives, wait,
in-place addition, correction and mask. All numerical premises are individual
operation contracts or transmitted peer inputs; no final task value is assumed. -/
theorem aggregation_values {α σ : Type} [Inhabited α] (obs : Observations α σ)
    (A : TensorArithmetic α) (rt : Runtime α σ) (lib : Library obs rt) (ops : TensorLibrary obs A rt)
    (gs gr : Nat → Nat) (comms : CollectiveLibrary obs A rt gs gr)
    (o : Options) (world rows count : Nat) (labels losses lse : Tensor)
    (initial : World α σ) (returned : Tensor × Tensor) (final : World α σ)
    (lv sv lossPeers lsePeers : Nat → α) (targets : Nat → Int)
    (loss : FloatArray obs initial losses (if count>1 then [count,rows] else [rows]) lv)
    (saved : FloatArray obs initial lse (if count>1 then [count,rows] else [rows]) sv)
    (label : IntArray obs initial labels [rows] targets)
    (separate : losses.region≠lse.region) (labelsSeparate : losses.region≠labels.region)
    (size : world>1 → world=gs (o.processGroup.getD 0))
    (e : AggregateData rt o world rows count labels losses lse initial returned final)
    (inputs : RankInputs comms e.rankCalls lossPeers lsePeers) :
    returned.1=e.localPair.1 ∧ Mutation obs initial final e.localPair.1 ∧
      FloatArray obs final returned.1 [rows]
        (aggregateLossValue A o world count rows lv sv lossPeers lsePeers targets) ∧
      FloatArray obs final returned.2 [rows]
        (rankResult A.logsumexp world rows (localResult A.logsumexp count rows sv) lsePeers) ∧
      IntArray obs final labels [rows] targets ∧ returned.1.region≠returned.2.region := by
  have localValues := local_reduction_values obs A rt ops count rows losses lse initial e.localPair e.localState
    lv sv loss saved separate e.localCalls
  have localSeparate := local_loss_separate obs A rt ops count rows losses lse initial e.localPair e.localState
    lv sv loss saved labels.region label.live labelsSeparate e.localCalls
  have ranked := rank_reduction_values obs A rt lib ops gs gr comms o world rows e.localPair e.rankPair
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
  have corrected := correction_values obs A rt ops o.squareScale rows e.addedLoss e.rankPair.2 e.correctedLoss e.added e.corrected
    _ rankLSE added lseAdded (by simpa only [addition.1] using ranked.2.2.2.2.symm) e.correctionCalls
  have labelCorrected := labelAdded.mutate corrected.2.1 (by simpa only [addition.1] using labelRankSeparate)
  have masked := mask_values obs A rt ops rows o.ignored e.correctedLoss labels e.mask returned.1
    e.corrected e.compared final _ targets corrected.2.2.1 labelCorrected e.compareCall e.maskCall
  have outputIdentity : returned.1=e.localPair.1 := masked.1.trans (corrected.1.trans (addition.1.trans ranked.1))
  have correctionDst : e.addedLoss=e.localPair.1 := addition.1.trans ranked.1
  have maskDst : e.correctedLoss=e.localPair.1 := corrected.1.trans correctionDst
  have allMutation : Mutation obs initial final e.localPair.1 := by
    have a : Mutation obs e.reduced e.added e.localPair.1 := by simpa only [ranked.1] using addition.2.1
    have c : Mutation obs e.added e.corrected e.localPair.1 := by simpa only [correctionDst] using corrected.2.1
    have m : Mutation obs e.corrected final e.localPair.1 := by simpa only [maskDst] using masked.2.1
    exact (((ranked.2.1.prepend localValues.frame).trans a).trans c).trans m
  have lseFinal := corrected.2.2.2.mutate masked.2.1
    (by simpa only [corrected.1,addition.1] using ranked.2.2.2.2.symm)
  have labelFinal := labelCorrected.mutate masked.2.1
    (by simpa only [maskDst] using localSeparate.symm)
  refine ⟨outputIdentity,allMutation,?_,?_,labelFinal,?_⟩
  · exact masked.2.2
  · simpa only [e.lseIdentity] using lseFinal
  · simpa only [masked.1,corrected.1,addition.1,e.lseIdentity] using ranked.2.2.2.2

#axiomsClean local_loss_separate
#axiomsClean aggregation_values
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Host
