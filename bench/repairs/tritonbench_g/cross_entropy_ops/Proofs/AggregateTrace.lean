import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.Trace
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.AggregateTrace

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Host
open CrossEntropy1.Host (World Step bind LocalTrace RankTrace)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

def localStage {α σ : Type} (rt : Runtime α σ) (count : Nat) (losses lse : Tensor)
    (s : World α σ) : Step (World α σ) (Tensor × Tensor) :=
  if count>1 then
    bind (rt.logsumexp0 lse s) fun l sl =>
    bind (rt.sum0 losses sl) fun v ss => .ok (v,l) ss
  else .ok (losses,lse) s

def rankStage {α σ : Type} (rt : Runtime α σ) (o : Options) (world rows : Nat)
    (pair : Tensor × Tensor) (s : World α σ) : Step (World α σ) (Tensor × Tensor) :=
  if world>1 then
    bind (rt.empty [world,rows] pair.2.dtype pair.2.device s) fun gathered allocated =>
    bind (rt.allGather gathered pair.2 (o.processGroup.getD 0) allocated) fun _ collected =>
    bind (rt.allReduceAsync pair.1 (o.processGroup.getD 0) collected) fun handle pending =>
    bind (rt.logsumexp0 gathered pending) fun globalLSE reduced =>
    bind (rt.wait handle reduced) fun _ waited => .ok (pair.1,globalLSE) waited
  else .ok pair s

def correctionStage {α σ : Type} (rt : Runtime α σ) (o : Options) (labels losses lse : Tensor)
    (s : World α σ) : Step (World α σ) (Tensor × Tensor) :=
  if o.squareScale≠0 then
    bind (rt.square lse s) fun squared ss =>
    bind (rt.scalarMul o.squareScale squared ss) fun scaled sm =>
    bind (rt.equalInt labels o.ignored sm) fun mask compared =>
    bind (rt.maskedFillZero scaled mask compared) fun zlosses masked =>
    bind (rt.addInplace losses zlosses masked) fun corrected sc => .ok (corrected,zlosses) sc
  else bind (rt.zerosLike losses s) fun zeros sz => .ok (losses,zeros) sz

theorem local_stage_trace {α σ : Type} (rt : Runtime α σ) (count : Nat) (losses lse : Tensor)
    (initial final : World α σ) (pair : Tensor × Tensor)
    (success : localStage rt count losses lse initial=.ok pair final) :
    LocalTrace rt.toRuntime count losses lse initial pair final := by
  by_cases many : count>1
  · simp only [localStage,if_pos many] at success
    obtain ⟨l,sl,hl,success⟩ := bind_success _ _ _ _ success
    obtain ⟨v,ss,hv,success⟩ := bind_success _ _ _ _ success
    simp only [Step.ok.injEq] at success
    obtain ⟨rfl,rfl⟩ := success
    exact .reduced many l v sl ss hl hv
  · simp only [localStage,if_neg many,Step.ok.injEq] at success
    obtain ⟨rfl,rfl⟩ := success
    exact .direct many

theorem rank_stage_trace {α σ : Type} (rt : Runtime α σ) (o : Options) (world rows : Nat)
    (input : Tensor × Tensor) (initial final : World α σ) (pair : Tensor × Tensor)
    (success : rankStage rt o world rows input initial=.ok pair final) :
    Nonempty (RankTrace rt.toRuntime o.toOptions world rows input initial pair final) := by
  by_cases many : world>1
  · simp only [rankStage,if_pos many] at success
    obtain ⟨gathered,allocated,ha,success⟩ := bind_success _ _ _ _ success
    obtain ⟨u,collected,hg,success⟩ := bind_success _ _ _ _ success
    cases u
    obtain ⟨handle,pending,hs,success⟩ := bind_success _ _ _ _ success
    obtain ⟨globalLSE,reduced,hl,success⟩ := bind_success _ _ _ _ success
    obtain ⟨u,waited,hw,success⟩ := bind_success _ _ _ _ success
    cases u
    simp only [Step.ok.injEq] at success
    obtain ⟨rfl,rfl⟩ := success
    exact ⟨.reduced many gathered globalLSE handle allocated collected pending reduced waited ha hg hs hl hw⟩
  · simp only [rankStage,if_neg many,Step.ok.injEq] at success
    obtain ⟨rfl,rfl⟩ := success
    exact ⟨.direct many⟩

inductive CorrectionTrace {α σ : Type} (rt : Runtime α σ) (o : Options)
    (labels losses lse : Tensor) (initial : World α σ) : Tensor × Tensor → World α σ → Prop
  | zero (zero : o.squareScale=0) (z : Tensor) (after : World α σ)
      (call : rt.zerosLike losses initial=.ok z after) :
      CorrectionTrace rt o labels losses lse initial (losses,z) after
  | nonzero (nonzero : o.squareScale≠0) (squared scaled mask z output : Tensor)
      (ss sm compared masked after : World α σ)
      (squareCall : rt.square lse initial=.ok squared ss)
      (scaleCall : rt.scalarMul o.squareScale squared ss=.ok scaled sm)
      (compareCall : rt.equalInt labels o.ignored sm=.ok mask compared)
      (maskCall : rt.maskedFillZero scaled mask compared=.ok z masked)
      (addCall : rt.addInplace losses z masked=.ok output after) :
      CorrectionTrace rt o labels losses lse initial (output,z) after

theorem correction_stage_trace {α σ : Type} (rt : Runtime α σ) (o : Options) (labels losses lse : Tensor)
    (initial final : World α σ) (pair : Tensor × Tensor)
    (success : correctionStage rt o labels losses lse initial=.ok pair final) :
    CorrectionTrace rt o labels losses lse initial pair final := by
  by_cases nonzero : o.squareScale≠0
  · simp only [correctionStage,if_pos nonzero] at success
    obtain ⟨sq,ss,hs,success⟩ := bind_success _ _ _ _ success
    obtain ⟨mul,sm,hm,success⟩ := bind_success _ _ _ _ success
    obtain ⟨mask,compared,hc,success⟩ := bind_success _ _ _ _ success
    obtain ⟨z,masked,hz,success⟩ := bind_success _ _ _ _ success
    obtain ⟨out,after,ha,success⟩ := bind_success _ _ _ _ success
    simp only [Step.ok.injEq] at success
    obtain ⟨rfl,rfl⟩ := success
    exact .nonzero nonzero sq mul mask z out ss sm compared masked after hs hm hc hz ha
  · simp only [correctionStage,if_neg nonzero] at success
    obtain ⟨z,after,hz,success⟩ := bind_success _ _ _ _ success
    simp only [Step.ok.injEq] at success
    obtain ⟨rfl,rfl⟩ := success
    exact .zero (not_not.mp nonzero) z after hz

structure AggregateData {α σ : Type} (rt : Runtime α σ) (o : Options) (world rows count : Nat)
    (labels losses lse : Tensor) (initial : World α σ) (returned : Aggregated) (final : World α σ) where
  localPair : Tensor × Tensor
  localState : World α σ
  localCalls : LocalTrace rt.toRuntime count losses lse initial localPair localState
  rankPair : Tensor × Tensor
  reduced : World α σ
  rankCalls : RankTrace rt.toRuntime o.toOptions world rows localPair localState rankPair reduced
  addedLoss : Tensor
  added : World α σ
  addCall : rt.addInplace rankPair.1 rankPair.2 reduced=.ok addedLoss added
  corrected : Tensor × Tensor
  correctedState : World α σ
  correctionCalls : CorrectionTrace rt o labels addedLoss rankPair.2 added corrected correctedState
  mask : Tensor
  compared : World α σ
  compareCall : rt.equalInt labels o.ignored correctedState=.ok mask compared
  maskCall : rt.maskedFillZero corrected.1 mask compared=.ok returned.losses final
  lseIdentity : returned.lse=rankPair.2
  zIdentity : returned.zLosses=corrected.2

/-- Extract every actual reduction, asynchronous wait, z-loss operation and
mask from successful execution. The CE1 local/rank trace types are reused. -/
theorem aggregate_success {α σ : Type} (rt : Runtime α σ) (o : Options) (world rows count : Nat)
    (labels losses lse : Tensor) (initial : World α σ) (returned : Aggregated) (final : World α σ)
    (success : aggregate rt o world rows count labels losses lse initial=.ok returned final) :
    Nonempty (AggregateData rt o world rows count labels losses lse initial returned final) := by
  change (bind (localStage rt count losses lse initial) fun localPair localState =>
    bind (rankStage rt o world rows localPair localState) fun pair reduced =>
    bind (rt.addInplace pair.1 pair.2 reduced) fun addedLoss added =>
    bind (correctionStage rt o labels addedLoss pair.2 added) fun corrected state =>
    bind (rt.equalInt labels o.ignored state) fun mask compared =>
    bind (rt.maskedFillZero corrected.1 mask compared) fun masked final =>
      .ok ⟨masked,pair.2,corrected.2⟩ final)=.ok returned final at success
  obtain ⟨lp,ls,hl,success⟩ := bind_success _ _ _ _ success
  obtain ⟨rp,rs,hr,success⟩ := bind_success _ _ _ _ success
  obtain ⟨added,sa,ha,success⟩ := bind_success _ _ _ _ success
  obtain ⟨corrected,sc,hc,success⟩ := bind_success _ _ _ _ success
  obtain ⟨mask,se,he,success⟩ := bind_success _ _ _ _ success
  obtain ⟨masked,sf,hm,success⟩ := bind_success _ _ _ _ success
  simp only [Step.ok.injEq] at success
  obtain ⟨rfl,rfl⟩ := success
  obtain ⟨ranked⟩ := rank_stage_trace rt o world rows lp ls rs rp hr
  exact ⟨⟨lp,ls,local_stage_trace rt count losses lse initial ls lp hl,rp,rs,ranked,
    added,sa,ha,corrected,sc,correction_stage_trace rt o labels added rp.2 sa sc corrected hc,
    mask,se,he,hm,rfl,rfl⟩⟩

#axiomsClean local_stage_trace
#axiomsClean rank_stage_trace
#axiomsClean correction_stage_trace
#axiomsClean aggregate_success
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Host
