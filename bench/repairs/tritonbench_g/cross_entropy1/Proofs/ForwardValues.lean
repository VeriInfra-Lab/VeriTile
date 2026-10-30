import bench.repairs.tritonbench_g.cross_entropy1.Proofs.ForwardPreservation
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.AggregationValues

namespace VeriTile.Bench.Repairs.CrossEntropy1.Host
open VeriTile.Triton Source
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 6000000

theorem Materialized.append {α σ : Type} [Inhabited α] {obs : Observations α σ}
    {input output : Tensor} {a b c : World α σ} (mat : Materialized obs input output a b)
    (frame : Preserves obs b c) : Materialized obs input output a c := by
  refine ⟨mat.metadata,mat.frame.trans frame,frame.live _ mat.live,?_,?_,mat.rank⟩
  · intro dtype i hi
    exact (frame.floatValue output mat.live i).trans (mat.floats dtype i hi)
  · intro dtype i hi
    exact (frame.integers output i mat.live).trans (mat.integers dtype i hi)

theorem local_loss_fresh {α σ : Type} [Inhabited α] (obs : Observations α σ)
    (A : TensorArithmetic α) (rt : Runtime α σ) (ops : TensorLibrary obs A rt)
    (count rows : Nat) (losses lse : Tensor) (old initial : World α σ) (pair : Tensor × Tensor) (final : World α σ)
    (lv sv : Nat → α)
    (loss : FloatArray obs initial losses (if count>1 then [count,rows] else [rows]) lv)
    (saved : FloatArray obs initial lse (if count>1 then [count,rows] else [rows]) sv)
    (frame : Preserves obs old initial) (fresh : ¬old.live losses.region)
    (trace : LocalTrace rt count losses lse initial pair final) : ¬old.live pair.1.region := by
  cases trace with
  | direct => exact fresh
  | reduced many localLSE localLoss logged summed lseCall sumCall =>
    have si : lse.shape=[count,rows] := by simpa only [if_pos many] using saved.shapeEq
    have li : losses.shape=[count,rows] := by simpa only [if_pos many] using loss.shapeEq
    have first := ops.logsumexp lse count rows initial localLSE logged saved.live si saved.dtype lseCall
    have second := ops.sum losses count rows logged localLoss _ (first.allocation.frame.live _ loss.live) li loss.dtype sumCall
    intro live
    exact second.allocation.fresh (first.allocation.frame.live _ (frame.live _ live))

inductive ForwardAggregation {α σ : Type} [Inhabited α] {M : FP32 α} {rt : Runtime α σ}
    {o : Options} {logits labels : Tensor} {initial final : World α σ} {returned : ForwardReturn}
    (e : ForwardData M rt o logits labels initial returned final) : Type
  | direct (flag : ¬(e.world>1 ∨ e.cols>65536)) (pair : e.pair=(e.losses,e.lse)) (state : e.aggregated=e.computed)
  | split (flag : e.world>1 ∨ e.cols>65536)
      (trace : AggregateData rt o e.world e.rows e.count e.labels e.losses e.lse e.computed e.pair e.aggregated)

theorem forward_aggregation {α σ : Type} [Inhabited α] {M : FP32 α} {rt : Runtime α σ}
    {o : Options} {logits labels : Tensor} {initial final : World α σ} {returned : ForwardReturn}
    (e : ForwardData M rt o logits labels initial returned final) : Nonempty (ForwardAggregation e) := by
  have success := e.aggregateCall
  by_cases flag : e.world>1 ∨ e.cols>65536
  · simp only [decide_eq_true flag,↓reduceIte] at success
    obtain ⟨trace⟩ := aggregate_success rt o e.world e.rows e.count e.labels e.losses e.lse e.computed e.pair e.aggregated success
    exact ⟨.split flag trace⟩
  · simp only [decide_eq_false flag,Bool.false_eq_true,↓reduceIte,Step.ok.injEq] at success
    exact ⟨.direct flag success.1.symm success.2.symm⟩

def ForwardInputs {α σ : Type} [Inhabited α] {obs : Observations α σ} {A : TensorArithmetic α}
    {rt : Runtime α σ} {gs gr : Nat → Nat} (comms : CollectiveLibrary obs A rt gs gr)
    {M : FP32 α} {o : Options} {logits labels : Tensor} {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (a : ForwardAggregation e)
    (lossPeers lsePeers : Nat → α) : Prop :=
  match a with
  | .direct _ _ _ => True
  | .split _ trace => RankInputs comms trace.rankCalls lossPeers lsePeers

def finalLoss {α σ : Type} [Inhabited α] {obs : Observations α σ} {M : FP32 α} {rt : Runtime α σ}
    {o : Options} {logits labels : Tensor} {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs e)
    (A : TensorArithmetic α) (lossPeers lsePeers : Nat → α) (row : Nat) : α :=
  if e.world>1 ∨ e.cols>65536 then
    aggregateLossValue A o e.world e.count e.rows k.lossValues k.lseValues lossPeers lsePeers (obs.integer initial labels) row
  else k.lossValues row

def finalLSE {α σ : Type} [Inhabited α] {obs : Observations α σ} {M : FP32 α} {rt : Runtime α σ}
    {o : Options} {logits labels : Tensor} {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs e)
    (A : TensorArithmetic α) (lsePeers : Nat → α) (row : Nat) : α :=
  if e.world>1 ∨ e.cols>65536 then
    rankResult A.logsumexp e.world e.rows (localResult A.logsumexp e.count e.rows k.lseValues) lsePeers row
  else k.lseValues row

/-- Actual forward result and saved data, for either wrapper branch. The
source-store functions are derived by ForwardKernelData; peer sends remain
explicitly linked through ForwardInputs for the whole-group theorem. -/
theorem forward_values {α σ : Type} [Inhabited α] (obs : Observations α σ)
    (A : TensorArithmetic α) (rt : Runtime α σ) (lib : Library obs rt) (ops : TensorLibrary obs A rt)
    (ints : IntegerViews obs) (gs gr : Nat → Nat) (comms : CollectiveLibrary obs A rt gs gr)
    {M : FP32 α} {o : Options} {logits labels : Tensor} {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs e)
    (labelDtype : labels.dtype="torch.int64")
    (size : e.world>1 → e.world=gs (o.processGroup.getD 0))
    (aggregation : ForwardAggregation e) (lossPeers lsePeers : Nat → α)
    (inputs : ForwardInputs comms aggregation lossPeers lsePeers) :
    FloatArray obs final returned.losses [e.rows] (finalLoss k A lossPeers lsePeers) ∧
    FloatArray obs final returned.context.lse [e.rows] (finalLSE k A lsePeers) ∧
    IntArray obs final returned.context.labels [e.rows] (obs.integer initial labels) ∧
    Materialized obs logits returned.context.logits initial final ∧ Preserves obs initial final := by
  have arrays := forward_arrays obs rt lib k
  have label := forward_label_array obs rt lib ints k labelDtype
  have computedFrame := forward_resolved_frame obs rt lib ints k
  have save := lib.save _ _ _ _ _ _ e.saveCall
  have returns := e.returnIdentity
  have lossReturn : returned.losses=e.pair.1 := congrArg ForwardReturn.losses returns
  have lseReturn : returned.context.lse=e.pair.2 := congrArg (fun r : ForwardReturn => r.context.lse) returns
  have labelsReturn : returned.context.labels=e.labels := congrArg (fun r : ForwardReturn => r.context.labels) returns
  have logitsReturn : returned.context.logits=e.workingLogits := congrArg (fun r : ForwardReturn => r.context.logits) returns
  rw [lossReturn,lseReturn,labelsReturn,logitsReturn]
  cases aggregation with
  | direct flag pair state =>
    have count : e.count=1 := one_block e.cols (fwdWidth e.cols) (forward_positive e).1
      ((forward_width e.cols (forward_positive e).1).2.2 (by omega))
    have losses : FloatArray obs e.computed e.losses [e.rows] k.lossValues := by simpa [count] using arrays.1
    have lses : FloatArray obs e.computed e.lse [e.rows] k.lseValues := by simpa [count] using arrays.2
    have endFrame : Preserves obs e.computed final := by simpa only [state] using save
    have whole := computedFrame.trans endFrame
    refine ⟨?_,?_,?_,k.working.append whole,k.working.frame.trans whole⟩
    · have eq : finalLoss k A lossPeers lsePeers=k.lossValues := by funext row; simp [finalLoss,flag]
      simpa only [pair,eq] using losses.preserve endFrame
    · have eq : finalLSE k A lsePeers=k.lseValues := by funext row; simp [finalLSE,flag]
      simpa only [pair,eq] using lses.preserve endFrame
    · exact label.preserve endFrame
  | split flag trace =>
    have aggregated := aggregation_values obs A rt lib ops gs gr comms o e.world e.rows e.count e.labels e.losses e.lse
      e.computed e.pair e.aggregated k.lossValues k.lseValues lossPeers lsePeers (obs.integer initial labels)
      arrays.1 arrays.2 label k.separation.outputDistinct.symm
      (by simpa only [ForwardData.parameters,forwardParameters,Region.cast_cast,Region.cast_id] using k.separation.lossLabels)
      size trace inputs
    have fresh := local_loss_fresh obs A rt ops e.count e.rows e.losses e.lse e.resolved e.computed trace.localPair trace.localState
      k.lossValues k.lseValues arrays.1 arrays.2 computedFrame k.lossAllocation.fresh trace.localCalls
    have whole := (computedFrame.freshMutation aggregated.2.1 fresh).trans save
    refine ⟨?_,?_,aggregated.2.2.2.2.1.preserve save,k.working.append whole,k.working.frame.trans whole⟩
    · have eq : finalLoss k A lossPeers lsePeers=aggregateLossValue A o e.world e.count e.rows k.lossValues k.lseValues lossPeers lsePeers (obs.integer initial labels) := by
        funext row; simp [finalLoss,flag]
      simpa only [eq] using aggregated.2.2.1.preserve save
    · have eq : finalLSE k A lsePeers=rankResult A.logsumexp e.world e.rows (localResult A.logsumexp e.count e.rows k.lseValues) lsePeers := by
        funext row; simp [finalLSE,flag]
      simpa only [eq] using aggregated.2.2.2.1.preserve save

#axiomsClean forward_aggregation
#axiomsClean forward_values
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Host
