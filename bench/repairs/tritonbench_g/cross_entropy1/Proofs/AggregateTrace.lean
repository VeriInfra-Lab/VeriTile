import bench.repairs.tritonbench_g.cross_entropy1.Proofs.ForwardKernel

namespace VeriTile.Bench.Repairs.CrossEntropy1.Host
open VeriTile.Triton Source
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

inductive LocalTrace {α σ : Type} (rt : Runtime α σ) (count : Nat)
    (losses lse : Tensor) (initial : World α σ) : Tensor × Tensor → World α σ → Prop
  | direct (one : ¬count>1) : LocalTrace rt count losses lse initial (losses,lse) initial
  | reduced (many : count>1) (localLSE localLoss : Tensor) (logged summed : World α σ)
      (lseCall : rt.logsumexp0 lse initial=.ok localLSE logged)
      (sumCall : rt.sum0 losses logged=.ok localLoss summed) :
      LocalTrace rt count losses lse initial (localLoss,localLSE) summed

inductive RankTrace {α σ : Type} (rt : Runtime α σ) (o : Options) (world rows : Nat)
    (localPair : Tensor × Tensor) (initial : World α σ) : Tensor × Tensor → World α σ → Type
  | direct (one : ¬world>1) : RankTrace rt o world rows localPair initial localPair initial
  | reduced (many : world>1) (gathered globalLSE : Tensor) (handle : Nat)
      (allocated collected pending reduced waited : World α σ)
      (allocationCall : rt.empty [world,rows] localPair.2.dtype localPair.2.device initial=.ok gathered allocated)
      (gatherCall : rt.allGather gathered localPair.2 (o.processGroup.getD 0) allocated=.ok () collected)
      (sumCall : rt.allReduceAsync localPair.1 (o.processGroup.getD 0) collected=.ok handle pending)
      (lseCall : rt.logsumexp0 gathered pending=.ok globalLSE reduced)
      (waitCall : rt.wait handle reduced=.ok () waited) :
      RankTrace rt o world rows localPair initial (localPair.1,globalLSE) waited

inductive CorrectionTrace {α σ : Type} (rt : Runtime α σ) (scale : ℝ)
    (losses lse : Tensor) (initial : World α σ) : Tensor → World α σ → Prop
  | zero (zero : scale=0) : CorrectionTrace rt scale losses lse initial losses initial
  | nonzero (nonzero : scale≠0) (squared scaled output : Tensor) (ss sm final : World α σ)
      (squareCall : rt.square lse initial=.ok squared ss)
      (scaleCall : rt.scalarMul scale squared ss=.ok scaled sm)
      (addCall : rt.addInplace losses scaled sm=.ok output final) :
      CorrectionTrace rt scale losses lse initial output final

structure AggregateData {α σ : Type} (rt : Runtime α σ) (o : Options) (world rows count : Nat)
    (labels losses lse : Tensor) (initial : World α σ) (returned : Tensor × Tensor) (final : World α σ) where
  localPair : Tensor × Tensor
  localState : World α σ
  localCalls : LocalTrace rt count losses lse initial localPair localState
  rankPair : Tensor × Tensor
  reduced : World α σ
  rankCalls : RankTrace rt o world rows localPair localState rankPair reduced
  addedLoss : Tensor
  added : World α σ
  addCall : rt.addInplace rankPair.1 rankPair.2 reduced=.ok addedLoss added
  correctedLoss : Tensor
  corrected : World α σ
  correctionCalls : CorrectionTrace rt o.squareScale addedLoss rankPair.2 added correctedLoss corrected
  mask : Tensor
  compared : World α σ
  compareCall : rt.equalInt labels o.ignored corrected=.ok mask compared
  maskCall : rt.maskedFillZero correctedLoss mask compared=.ok returned.1 final
  lseIdentity : returned.2=rankPair.2

theorem aggregate_success {α σ : Type} (rt : Runtime α σ) (o : Options) (world rows count : Nat)
    (labels losses lse : Tensor) (initial : World α σ) (returned : Tensor × Tensor) (final : World α σ)
    (success : aggregate rt o world rows count labels losses lse initial=.ok returned final) :
    Nonempty (AggregateData rt o world rows count labels losses lse initial returned final) := by
  let localStage := if count>1 then
    bind (rt.logsumexp0 lse initial) fun nextLSE sl =>
      bind (rt.sum0 losses sl) fun nextLoss ss => .ok (nextLoss,nextLSE) ss
    else .ok (losses,lse) initial
  have localTrace : ∀ pair state, localStage=.ok pair state → LocalTrace rt count losses lse initial pair state := by
    intro pair state h
    by_cases many : count>1
    · simp only [localStage,if_pos many] at h
      cases hl : rt.logsumexp0 lse initial with
      | error reason failed => simp [bind,hl] at h
      | ok localLSE logged =>
        simp only [hl,bind] at h
        cases hs : rt.sum0 losses logged with
        | error reason failed => simp [bind,hs] at h
        | ok localLoss summed =>
          simp only [hs,bind,Step.ok.injEq] at h
          obtain ⟨rfl,rfl⟩ := h
          exact .reduced many localLSE localLoss logged summed hl hs
    · simp only [localStage,if_neg many,Step.ok.injEq] at h
      obtain ⟨rfl,rfl⟩ := h
      exact .direct many
  let ranks := fun (pair : Tensor × Tensor) state => if world>1 then
    bind (rt.empty [world,rows] pair.2.dtype pair.2.device state) fun gathered allocated =>
    bind (rt.allGather gathered pair.2 (o.processGroup.getD 0) allocated) fun _ collected =>
    bind (rt.allReduceAsync pair.1 (o.processGroup.getD 0) collected) fun handle pending =>
    bind (rt.logsumexp0 gathered pending) fun globalLSE reduced =>
    bind (rt.wait handle reduced) fun _ waited => .ok (pair.1,globalLSE) waited
    else .ok pair state
  have rankTrace : ∀ input before pair state, ranks input before=.ok pair state →
      RankTrace rt o world rows input before pair state := by
    intro input before pair state h
    by_cases many : world>1
    · simp only [ranks,if_pos many] at h
      cases ha : rt.empty [world,rows] input.2.dtype input.2.device before with
      | error reason failed => simp [bind,ha] at h
      | ok gathered allocated =>
        simp only [ha,bind] at h
        cases hg : rt.allGather gathered input.2 (o.processGroup.getD 0) allocated with
        | error reason failed => simp [bind,hg] at h
        | ok u collected =>
          cases u
          simp only [hg,bind] at h
          cases hs : rt.allReduceAsync input.1 (o.processGroup.getD 0) collected with
          | error reason failed => simp [bind,hs] at h
          | ok handle pending =>
            simp only [hs,bind] at h
            cases hl : rt.logsumexp0 gathered pending with
            | error reason failed => simp [bind,hl] at h
            | ok globalLSE reduced =>
              simp only [hl,bind] at h
              cases hw : rt.wait handle reduced with
              | error reason failed => simp [bind,hw] at h
              | ok u waited =>
                cases u
                simp only [hw,bind,Step.ok.injEq] at h
                obtain ⟨rfl,rfl⟩ := h
                exact .reduced many gathered globalLSE handle allocated collected pending reduced waited ha hg hs hl hw
    · simp only [ranks,if_neg many,Step.ok.injEq] at h
      obtain ⟨rfl,rfl⟩ := h
      exact .direct many
  let correction := fun losses lse state => if o.squareScale≠0 then
    bind (rt.square lse state) fun squared ss =>
    bind (rt.scalarMul o.squareScale squared ss) fun scaled sm => rt.addInplace losses scaled sm
    else .ok losses state
  have correctionTrace : ∀ losses lse before output state, correction losses lse before=.ok output state →
      CorrectionTrace rt o.squareScale losses lse before output state := by
    intro losses lse before output state h
    by_cases nonzero : o.squareScale≠0
    · simp only [correction,if_pos nonzero] at h
      cases hs : rt.square lse before with
      | error reason failed => simp [bind,hs] at h
      | ok squared ss =>
        simp only [hs,bind] at h
        cases hm : rt.scalarMul o.squareScale squared ss with
        | error reason failed => simp [bind,hm] at h
        | ok scaled sm =>
          simp only [hm,bind] at h
          exact .nonzero nonzero squared scaled output ss sm state hs hm h
    · simp only [correction,if_neg nonzero,Step.ok.injEq] at h
      obtain ⟨rfl,rfl⟩ := h
      exact .zero (not_not.mp nonzero)
  change (bind localStage fun localPair localState => bind (ranks localPair localState) fun pair reduced =>
    bind (rt.addInplace pair.1 pair.2 reduced) fun addedLoss added =>
    bind (correction addedLoss pair.2 added) fun correctedLoss corrected =>
    bind (rt.equalInt labels o.ignored corrected) fun mask compared =>
    bind (rt.maskedFillZero correctedLoss mask compared) fun masked closed =>
      .ok (masked,pair.2) closed)=.ok returned final at success
  cases hl : localStage with
  | error reason failed => simp [bind,hl] at success
  | ok localPair localState =>
    simp only [hl,bind] at success
    cases hr : ranks localPair localState with
    | error reason failed => simp [bind,hr] at success
    | ok pair reduced =>
      simp only [hr,bind] at success
      cases ha : rt.addInplace pair.1 pair.2 reduced with
      | error reason failed => simp [bind,ha] at success
      | ok addedLoss added =>
        simp only [ha,bind] at success
        cases hc : correction addedLoss pair.2 added with
        | error reason failed => simp [bind,hc] at success
        | ok correctedLoss corrected =>
          simp only [hc,bind] at success
          cases he : rt.equalInt labels o.ignored corrected with
          | error reason failed => simp [bind,he] at success
          | ok mask compared =>
            simp only [he,bind] at success
            cases hm : rt.maskedFillZero correctedLoss mask compared with
            | error reason failed => simp [bind,hm] at success
            | ok masked closed =>
              simp only [hm,bind,Step.ok.injEq] at success
              obtain ⟨rfl,rfl⟩ := success
              exact ⟨⟨localPair,localState,localTrace _ _ hl,pair,reduced,rankTrace _ _ _ _ hr,
                addedLoss,added,ha,correctedLoss,corrected,correctionTrace _ _ _ _ _ hc,
                mask,compared,he,hm,rfl⟩⟩

#axiomsClean aggregate_success
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Host
