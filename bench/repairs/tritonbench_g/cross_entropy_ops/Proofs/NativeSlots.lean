import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.NativeCalls

namespace VeriTile.Bench.Repairs.CrossEntropyOps.NativeGraph
open Host
open CrossEntropy1.Floating
open CrossEntropy1.Host (World Observations Step groupSize groupRank lastStride snapshot_success LocalTrace RankTrace)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

/-- Source slots are structural positions in the extracted successful trace,
not arbitrary successful calls on an admitted runtime. -/
def LocalMember {σ : Type} {rt : Runtime Expr σ} {count losses lse initial pair final}
    (_trace : LocalTrace rt.toRuntime count losses lse initial pair final) (slot : List Nat) (c : SourceCall σ) : Prop :=
  count>1 ∧ ∃ (logged : World Expr σ)
    (hl : rt.logsumexp0 lse initial=.ok pair.2 logged)
    (hs : rt.sum0 losses logged=.ok pair.1 final),
      (slot=[0] ∧ c=SourceCall.lse rt lse pair.2 initial logged hl) ∨
      (slot=[1] ∧ c=SourceCall.sum rt losses pair.1 logged final hs)

def CorrectionMember {σ : Type} {rt : Runtime Expr σ} {o labels losses lse initial pair final}
    (_trace : CorrectionTrace rt o labels losses lse initial pair final) (slot : List Nat) (c : SourceCall σ) : Prop :=
  (o.squareScale=0 ∧ pair.1=losses ∧ ∃ call : rt.zerosLike losses initial=.ok pair.2 final,
    slot=[0] ∧ c=SourceCall.zeros rt losses pair.2 initial final call) ∨
  (o.squareScale≠0 ∧ ∃ (squared scaled mask : Tensor) (ss sm compared masked : World Expr σ)
    (hs : rt.square lse initial=.ok squared ss)
    (hm : rt.scalarMul o.squareScale squared ss=.ok scaled sm)
    (hc : rt.equalInt labels o.ignored sm=.ok mask compared)
    (hz : rt.maskedFillZero scaled mask compared=.ok pair.2 masked)
    (ha : rt.addInplace losses pair.2 masked=.ok pair.1 final),
      (slot=[0] ∧ c=SourceCall.square rt lse squared initial ss hs) ∨
      (slot=[1] ∧ c=SourceCall.scale rt o.squareScale squared scaled ss sm hm) ∨
      (slot=[2] ∧ c=SourceCall.compare rt labels o.ignored mask sm compared hc) ∨
      (slot=[3] ∧ c=SourceCall.mask rt scaled mask pair.2 compared masked hz) ∨
      (slot=[4] ∧ c=SourceCall.add rt losses pair.2 pair.1 masked final ha))

/-- The wait record retains this exact start → pending → LSE → reduced → wait
chain. A reused numeric handle from another start does not produce this slot. -/
def RankMember {σ : Type} {rt : Runtime Expr σ} {o : Options} {world rows pair initial output final}
    (trace : RankTrace rt.toRuntime o.toOptions world rows pair initial output final) (slot : List Nat) (c : SourceCall σ) : Prop :=
  match trace with
  | .direct _ => False
  | .reduced _ gathered globalLSE handle allocated collected pending reduced waited ha hg hs hl hw =>
      (slot=[0] ∧ c=SourceCall.empty rt [world,rows] pair.2.dtype pair.2.device gathered initial allocated ha) ∨
      (slot=[1] ∧ c=⟨rt,.gather,allocated,collected,[pair.2],[gathered],pair.2.device,o.processGroup.getD 0,none,
        .collective pair.1 gathered globalLSE handle (o.processGroup.getD 0) collected pending reduced waited,
        .gather _ _ _ _ _ hg⟩) ∨
      (slot=[2] ∧ c=⟨rt,.reduceStart,collected,pending,[pair.1],[pair.1],pair.1.device,o.processGroup.getD 0,some handle,
        .collective pair.1 gathered globalLSE handle (o.processGroup.getD 0) collected pending reduced waited,
        .reduceStart _ _ _ _ _ hs⟩) ∨
      (slot=[3] ∧ c=SourceCall.lse rt gathered globalLSE pending reduced hl) ∨
      (slot=[4] ∧ c=⟨rt,.reduceWait,reduced,waited,[pair.1],[pair.1],pair.1.device,o.processGroup.getD 0,some handle,
        .collective pair.1 gathered globalLSE handle (o.processGroup.getD 0) collected pending reduced waited,
        .reduceWait _ _ _ _ _ _ _ hs hw⟩)

def AggregateMember {σ : Type} {rt : Runtime Expr σ} {o world rows count labels losses lse initial result final}
    (a : AggregateData rt o world rows count labels losses lse initial result final) (slot : List Nat) (c : SourceCall σ) : Prop :=
  (∃ rest, slot=0::rest ∧ LocalMember a.localCalls rest c) ∨
  (∃ rest, slot=1::rest ∧ RankMember a.rankCalls rest c) ∨
  (slot=[2] ∧ c=SourceCall.add rt a.rankPair.1 a.rankPair.2 a.addedLoss a.reduced a.added a.addCall) ∨
  (∃ rest, slot=3::rest ∧ CorrectionMember a.correctionCalls rest c) ∨
  (slot=[4] ∧ c=SourceCall.compare rt labels o.ignored a.mask a.correctedState a.compared a.compareCall) ∨
  (slot=[5] ∧ c=SourceCall.mask rt a.corrected.1 a.mask result.losses a.compared final a.maskCall)

def ForwardMember {σ : Type} {obs : Observations Expr σ} (call : Entry.Invocation obs algebra)
    (slot : List Nat) (c : SourceCall σ) : Prop :=
  let d := call.data
  let snap := Classical.choice (snapshot_success call.runtime.toRuntime call.labels d.labels d.ranked d.labelsReady d.snapshotCall)
  (∃ (g : Nat) (hg : call.options.processGroup=some g), slot=[0] ∧
    c=⟨call.runtime,.query,call.initial,d.sized,[],[],call.logits.device,g,none,.natural d.world,
      .worldSize g call.initial d.world d.sized (by simpa only [groupSize,hg] using d.sizeCall)⟩) ∨
  (∃ (g : Nat) (hg : call.options.processGroup=some g), slot=[1] ∧
    c=⟨call.runtime,.query,d.sized,d.ranked,[],[],call.logits.device,g,none,.natural d.rank,
      .rank g d.sized d.rank d.ranked (by simpa only [groupRank,hg] using d.rankCall)⟩) ∨
  (slot=[2,0] ∧ c=SourceCall.resolve call.runtime call.labels snap.resolved d.ranked snap.intermediate snap.resolveCall) ∨
  (slot=[2,1] ∧ c=SourceCall.clone call.runtime snap.resolved d.labels snap.intermediate d.labelsReady snap.cloneCall) ∨
  (∃ h : lastStride call.logits≠1, slot=[3] ∧ c=SourceCall.contiguous call.runtime call.logits d.inplaceLogits
    d.labelsReady d.normalized (by simpa only [if_pos h] using d.originalNormalize)) ∨
  (slot=[4] ∧ c=SourceCall.resolve call.runtime d.inplaceLogits d.resolvedLogits d.normalized d.resolvedState d.resolveCall) ∨
  (∃ h : lastStride d.resolvedLogits≠1, slot=[5] ∧ c=SourceCall.contiguous call.runtime d.resolvedLogits d.workingLogits
    d.resolvedState d.resolved (by simpa only [if_pos h] using d.workingNormalize)) ∨
  (slot=[6] ∧ c=SourceCall.empty call.runtime (if d.count>1 then [d.count,d.rows] else [d.rows]) "torch.float32"
    d.workingLogits.device d.losses d.resolved d.lossAllocated d.lossCall) ∨
  (slot=[7] ∧ c=SourceCall.empty call.runtime (if d.count>1 then [d.count,d.rows] else [d.rows]) "torch.float32"
    d.workingLogits.device d.lse d.lossAllocated d.lseAllocated d.lseCall) ∨
  (slot=[8] ∧ c=SourceCall.empty call.runtime (if d.count>1 then [d.count,d.rows] else [d.rows]) "torch.float32"
    d.workingLogits.device d.zlosses d.lseAllocated d.zAllocated d.zCall) ∨
  (slot=[9] ∧ c=SourceCall.forward call) ∨
  (∃ split rest, slot=10::rest ∧ AggregateMember (call.splitData split) rest c) ∨
  (slot=[11] ∧ c=⟨call.runtime,.save,d.aggregated,d.saved,[d.workingLogits,d.result.lse,d.labels,d.inplaceLogits],[],
    d.workingLogits.device,0,none,.plain,.save _ _ _ _ _ _ d.saveCall⟩) ∨
  (slot=[12] ∧ c=SourceCall.mark call.runtime d.result.zLosses d.saved call.saved d.markCall)

def BackwardMember {σ : Type} {rt : Runtime Expr σ} {context input output before after}
    (b : BackwardData algebra rt context input output before after) (slot : List Nat) (c : SourceCall σ) : Prop :=
  let snap := Classical.choice (snapshot_success rt.toRuntime input b.upstream before b.upstreamReady b.snapshotCall)
  (slot=[0,0] ∧ c=SourceCall.resolve rt input snap.resolved before snap.intermediate snap.resolveCall) ∨
  (slot=[0,1] ∧ c=SourceCall.clone rt snap.resolved b.upstream snap.intermediate b.upstreamReady snap.cloneCall) ∨
  (∃ h : context.options.inplace=Bool.false, slot=[1] ∧ c=SourceCall.emptyLike rt context.logits Bool.false b.first
    b.upstreamReady b.allocated (by simpa only [h,Bool.false_eq_true,if_false] using b.firstCall)) ∨
  (∃ h : lastStride b.first≠1, slot=[2] ∧ c=SourceCall.emptyLike rt context.logits Bool.true b.work
    b.allocated b.prepared (by simpa only [if_pos h] using b.workCall)) ∨
  (slot=[3] ∧ c=SourceCall.backward b) ∨
  (∃ (different : b.work.objectId≠output.objectId) (returned : Tensor) (copy : rt.copy output b.work b.computed=.ok returned after),
    slot=[4] ∧ c=SourceCall.copy rt output b.work returned b.computed after copy)

-- The participant/slot identity type is generic and reused unchanged.
abbrev Participant := CrossEntropy1.Floating.Participant
abbrev SourceLocation := CrossEntropy1.Floating.SourceLocation

#axiomsClean ForwardMember
#axiomsClean BackwardMember
#axiomsClean RankMember
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.NativeGraph
