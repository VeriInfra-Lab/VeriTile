import bench.repairs.tritonbench_g.cross_entropy2.Proofs.Arrays
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.AggregateTrace
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.AggregationValues
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.ForwardValues

namespace VeriTile.Bench.Repairs.CrossEntropy2.Host
open VeriTile.Triton Source
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Step Observations Preserves FloatArray Produced value localResult)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 6000000

def aggregateZValue {α : Type} (A : Arithmetic α) (o : Options) (L : Nat → α) (targets : Nat → Int) (row : Nat) : α :=
  if targets row=o.ignored then A.zero else A.multiply (A.scale o.squareScale (L row)) (L row)
def aggregateLossValue {α : Type} (A : Arithmetic α) (o : Options) (loss L : Nat → α) (targets : Nat → Int) (row : Nat) : α :=
  if targets row=o.ignored then A.zero else
    A.add (A.add (loss row) (L row)) (A.multiply (A.scale o.squareScale (L row)) (L row))

/-- Complete values in all three final objects follow from the actual local
calls. The new z-loss is separately masked and kept distinct from the LSE. -/
theorem aggregation_values {α σ : Type} [Inhabited α] (obs : Observations α σ) (A : Arithmetic α)
    (rt : Runtime α σ) (lib : Library obs A rt) (o : Options) (rows count : Nat)
    (labels losses lse : Tensor) (initial : World α σ) (returned : Return) (final : World α σ)
    (lv sv : Nat → α) (targets : Nat → Int)
    (loss : FloatArray obs initial losses (if count>1 then [count,rows] else [rows]) lv)
    (saved : FloatArray obs initial lse (if count>1 then [count,rows] else [rows]) sv)
    (label : IntegerArray obs "torch.int32" initial labels [rows] targets)
    (separate : losses.region≠lse.region) (labelsSeparate : losses.region≠labels.region)
    (e : AggregateData rt o count labels losses lse initial returned final) :
    returned.1=e.pair.1 ∧ returned.2.2=e.zLoss ∧
      FloatArray obs final returned.1 [rows]
        (aggregateLossValue A o (localResult A.sum count rows lv) (localResult A.logsumexp count rows sv) targets) ∧
      FloatArray obs final returned.2.1 [rows] (localResult A.logsumexp count rows sv) ∧
      FloatArray obs final returned.2.2 [rows] (aggregateZValue A o (localResult A.logsumexp count rows sv) targets) ∧
      IntegerArray obs "torch.int32" final labels [rows] targets ∧
      returned.1.region≠returned.2.1.region ∧ returned.1.region≠returned.2.2.region ∧
      returned.2.1.region≠returned.2.2.region ∧
      ∀ old, Preserves obs old initial → ¬old.live losses.region → Preserves obs old final := by
  let L := localResult A.logsumexp count rows sv
  let lossValue := localResult A.sum count rows lv
  have locals := VeriTile.Bench.Repairs.CrossEntropy1.Host.local_reduction_values obs A.toTensorArithmetic rt.toRuntime lib.operations
    count rows losses lse initial e.pair e.reduced lv sv loss saved separate e.localCalls
  have localSeparate := VeriTile.Bench.Repairs.CrossEntropy1.Host.local_loss_separate obs A.toTensorArithmetic rt.toRuntime lib.operations
    count rows losses lse initial e.pair e.reduced lv sv loss saved labels.region label.live labelsSeparate e.localCalls
  have labelsReduced := label.preserve locals.frame
  have addition := add_arrays obs A rt lib e.pair.1 e.pair.2 e.addedLoss e.reduced e.added [rows]
    lossValue L locals.losses locals.lse e.addCall
  have lseAdded := locals.lse.mutate addition.2.1 locals.separate.symm
  have labelsAdded := labelsReduced.mutate addition.2.1 localSeparate.symm
  have scale := scale_array obs A rt lib o.squareScale e.pair.2 e.scaled e.added e.scaledState [rows] L lseAdded e.scaleCall
  have lossScaled := addition.2.2.preserve scale.allocation.frame
  have lseScaled := lseAdded.preserve scale.allocation.frame
  have labelsScaled := labelsAdded.preserve scale.allocation.frame
  have multiplication := multiply_arrays obs A rt lib e.scaled e.pair.2 e.zLoss e.scaledState e.multiplied [rows]
    (fun i => A.scale o.squareScale (L i)) L scale.array lseScaled e.multiplyCall
  have zNeLoss : e.zLoss.region≠e.addedLoss.region := by
    intro equal; apply multiplication.allocation.fresh; simpa only [equal] using lossScaled.live
  have zNeLSE : e.zLoss.region≠e.pair.2.region := by
    intro equal; apply multiplication.allocation.fresh; simpa only [equal] using lseScaled.live
  have zNeLabels : e.zLoss.region≠labels.region := by
    intro equal; apply multiplication.allocation.fresh; simpa only [equal] using labelsScaled.live
  have lossMultiplied := lossScaled.preserve multiplication.allocation.frame
  have lseMultiplied := lseScaled.preserve multiplication.allocation.frame
  have labelsMultiplied := labelsScaled.preserve multiplication.allocation.frame
  have correction := add_arrays obs A rt lib e.addedLoss e.zLoss e.correctedLoss e.multiplied e.corrected [rows]
    (fun i => A.add (lossValue i) (L i)) (fun i => A.multiply (A.scale o.squareScale (L i)) (L i))
    lossMultiplied multiplication.array e.correctionCall
  have zCorrected := multiplication.array.mutate correction.2.1 zNeLoss
  have lseCorrected := lseMultiplied.mutate correction.2.1 (by simpa only [addition.1] using locals.separate.symm)
  have labelsCorrected := labelsMultiplied.mutate correction.2.1 (by simpa only [addition.1] using localSeparate.symm)
  have widened := widen_array obs A rt lib labels e.wideLabels e.corrected e.widened [rows] targets labelsCorrected e.widenCall
  have lossWide := correction.2.2.preserve widened.1.frame
  have zWide := zCorrected.preserve widened.1.frame
  have lseWide := lseCorrected.preserve widened.1.frame
  have labelsWide := labelsCorrected.preserve widened.1.frame
  have compared := compare_array obs A rt lib e.wideLabels e.mask o.ignored e.widened e.compared [rows] targets widened.2.2 e.compareCall
  have lossCompared := lossWide.preserve compared.1.frame
  have zCompared := zWide.preserve compared.1.frame
  have lseCompared := lseWide.preserve compared.1.frame
  have labelsCompared := labelsWide.preserve compared.1.frame
  have maskNeLoss : e.mask.region≠e.correctedLoss.region := by
    intro equal; apply compared.1.fresh; simpa only [equal] using lossWide.live
  have maskedLoss := mask_array obs A rt lib e.correctedLoss e.mask returned.1 e.compared e.masked [rows]
    (fun i => A.add (A.add (lossValue i) (L i)) (A.multiply (A.scale o.squareScale (L i)) (L i)))
    (fun i => decide (targets i=o.ignored)) lossCompared compared.2.2 e.maskLossCall
  have zMasked := zCompared.mutate maskedLoss.2.1 (by simpa only [correction.1] using zNeLoss)
  have boolMasked := compared.2.2.mutate maskedLoss.2.1 maskNeLoss
  have lseMasked := lseCompared.mutate maskedLoss.2.1 (by simpa only [correction.1,addition.1] using locals.separate.symm)
  have labelsMasked := labelsCompared.mutate maskedLoss.2.1 (by simpa only [correction.1,addition.1] using localSeparate.symm)
  have maskedZ := mask_array obs A rt lib e.zLoss e.mask returned.2.2 e.masked final [rows]
    (fun i => A.multiply (A.scale o.squareScale (L i)) (L i)) (fun i => decide (targets i=o.ignored))
    zMasked boolMasked e.maskZCall
  have lossFinal := maskedLoss.2.2.mutate maskedZ.2.1 (by simpa only [maskedLoss.1,correction.1] using zNeLoss.symm)
  have lseFinal := lseMasked.mutate maskedZ.2.1 zNeLSE.symm
  have labelsFinal := labelsMasked.mutate maskedZ.2.1 zNeLabels.symm
  refine ⟨maskedLoss.1.trans (correction.1.trans addition.1),maskedZ.1,?_,?_,?_,labelsFinal,?_,?_,?_,?_⟩
  · simpa only [aggregateLossValue,decide_eq_true_eq,L,lossValue] using lossFinal
  · simpa only [e.lseIdentity,L] using lseFinal
  · simpa only [aggregateZValue,decide_eq_true_eq,L] using maskedZ.2.2
  · simpa only [maskedLoss.1,correction.1,addition.1,e.lseIdentity] using locals.separate
  · simpa only [maskedLoss.1,correction.1,maskedZ.1] using zNeLoss.symm
  · simpa only [e.lseIdentity,maskedZ.1] using zNeLSE.symm
  · intro old oldFrame oldFresh
    have fresh := VeriTile.Bench.Repairs.CrossEntropy1.Host.local_loss_fresh obs A.toTensorArithmetic rt.toRuntime lib.operations
      count rows losses lse old initial e.pair e.reduced lv sv loss saved oldFrame oldFresh e.localCalls
    have addedFrame := (oldFrame.trans locals.frame).freshMutation addition.2.1 fresh
    have scaledFrame := addedFrame.trans scale.allocation.frame
    have zFresh : ¬old.live e.zLoss.region := fun live => multiplication.allocation.fresh (scaledFrame.live _ live)
    have multipliedFrame := scaledFrame.trans multiplication.allocation.frame
    have correctedFrame := multipliedFrame.freshMutation correction.2.1 (by simpa only [addition.1] using fresh)
    have comparedFrame := (correctedFrame.trans widened.1.frame).trans compared.1.frame
    have maskedFrame := comparedFrame.freshMutation maskedLoss.2.1 (by simpa only [correction.1,addition.1] using fresh)
    exact maskedFrame.freshMutation maskedZ.2.1 zFresh

#axiomsClean aggregation_values
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Host
