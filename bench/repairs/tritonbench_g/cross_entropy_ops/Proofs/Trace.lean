import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.Host
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Library

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Host
open VeriTile.Triton
open CrossEntropy1.Source (FP32 Memory)
open CrossEntropy1.Host (World Step rowStride lastStride fwdWidth bwdWidth warps)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

/-- Generic decomposition of the existing exception/state bind. The extracted
facts concern actual API calls, never desired output values. -/
theorem bind_success {S A B : Type} (first : Step S A) (next : A → S → Step S B)
    (value : B) (final : S) (success : CrossEntropy1.Host.bind first next=.ok value final) :
    ∃ intermediate state, first=.ok intermediate state ∧ next intermediate state=.ok value final := by
  cases first with
  | error detail state => cases success
  | ok intermediate state => exact ⟨intermediate,state,rfl,success⟩

abbrev ForwardLaunchData {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (device : Nat) (p : Parameters) (count : Nat) (before after : World α σ) :=
  CrossEntropy1.Host.LaunchData rt.toRuntime (rt.setupForwardOps p (warps p.block))
    (fun memory => Grid.forward M p memory count) device before after

abbrev BackwardLaunchData {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (device : Nat) (p : Parameters) (count : Nat) (before after : World α σ) :=
  CrossEntropy1.Host.LaunchData rt.toRuntime (rt.setupBackwardOps p (warps p.block))
    (fun memory => Grid.backward M p memory count) device before after

theorem forward_launch_success {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (device : Nat) (p : Parameters) (count : Nat) (before after : World α σ)
    (success : forwardLaunch M rt device p count before=.ok () after) :
    Nonempty (ForwardLaunchData M rt device p count before after) :=
  CrossEntropy1.Host.launch_success rt.toRuntime _ _ device before after success

theorem backward_launch_success {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (device : Nat) (p : Parameters) (count : Nat) (before after : World α σ)
    (success : backwardLaunch M rt device p count before=.ok () after) :
    Nonempty (BackwardLaunchData M rt device p count before after) :=
  CrossEntropy1.Host.launch_success rt.toRuntime _ _ device before after success

structure ForwardData {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (o : Options) (callerLogits callerLabels : Tensor) (initial : World α σ)
    (returned : ForwardReturn) (final : World α σ) where
  rows : Nat
  cols : Nat
  logitsShape : callerLogits.shape=[rows,cols]
  labelsShape : callerLabels.shape=[rows]
  positive : cols≠0
  world : Nat
  sized : World α σ
  sizeCall : CrossEntropy1.Host.groupSize rt.toRuntime o.processGroup initial=.ok world sized
  rank : Nat
  ranked : World α σ
  rankCall : CrossEntropy1.Host.groupRank rt.toRuntime o.processGroup sized=.ok rank ranked
  labels : Tensor
  labelsReady : World α σ
  snapshotCall : CrossEntropy1.Host.snapshot rt.toRuntime callerLabels ranked=.ok labels labelsReady
  inplaceLogits : Tensor
  normalized : World α σ
  originalNormalize : (if lastStride callerLogits≠1 then rt.contiguous callerLogits labelsReady else
    .ok callerLogits labelsReady)=Step.ok inplaceLogits normalized
  overlapGuard : ¬(o.inplace && decide (rows>1 ∧ rowStride inplaceLogits<cols))=Bool.true
  resolvedLogits : Tensor
  resolvedState : World α σ
  resolveCall : rt.resolveNeg inplaceLogits normalized=.ok resolvedLogits resolvedState
  workingLogits : Tensor
  resolved : World α σ
  workingNormalize : (if lastStride resolvedLogits≠1 then rt.contiguous resolvedLogits resolvedState else
    .ok resolvedLogits resolvedState)=Step.ok workingLogits resolved
  losses : Tensor
  lossAllocated : World α σ
  lossCall : rt.empty (if CrossEntropy1.Math.blocks cols (fwdWidth cols)>1 then
    [CrossEntropy1.Math.blocks cols (fwdWidth cols),rows] else [rows])
    "torch.float32" workingLogits.device resolved=.ok losses lossAllocated
  lse : Tensor
  lseAllocated : World α σ
  lseCall : rt.empty (if CrossEntropy1.Math.blocks cols (fwdWidth cols)>1 then
    [CrossEntropy1.Math.blocks cols (fwdWidth cols),rows] else [rows])
    "torch.float32" workingLogits.device lossAllocated=.ok lse lseAllocated
  zlosses : Tensor
  zAllocated : World α σ
  zCall : rt.empty (if CrossEntropy1.Math.blocks cols (fwdWidth cols)>1 then
    [CrossEntropy1.Math.blocks cols (fwdWidth cols),rows] else [rows])
    "torch.float32" workingLogits.device lseAllocated=.ok zlosses zAllocated
  computed : World α σ
  launchCall : forwardLaunch M rt workingLogits.device
    (forwardParameters o workingLogits labels losses lse zlosses rows cols (world*cols) (rank*cols)
      (fwdWidth cols) (decide (world>1 ∨ cols>65536))) (CrossEntropy1.Math.blocks cols (fwdWidth cols))
      zAllocated=.ok () computed
  result : Aggregated
  aggregated : World α σ
  aggregateCall : (if decide (world>1 ∨ cols>65536) then
    aggregate rt o world rows (CrossEntropy1.Math.blocks cols (fwdWidth cols)) labels losses lse computed
    else .ok ⟨losses,lse,zlosses⟩ computed)=Step.ok result aggregated
  saved : World α σ
  saveCall : rt.saveTensors workingLogits result.lse labels inplaceLogits aggregated=.ok () saved
  markCall : rt.markNonDifferentiable result.zLosses saved=.ok () final
  returnIdentity : returned=⟨result.losses,result.zLosses,
    ⟨workingLogits,result.lse,labels,inplaceLogits,o,world*cols,rank*cols⟩⟩

theorem forward_success {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (o : Options) (callerLogits callerLabels : Tensor) (initial : World α σ)
    (returned : ForwardReturn) (final : World α σ)
    (success : forward M rt o callerLogits callerLabels initial=.ok returned final) :
    Nonempty (ForwardData M rt o callerLogits callerLabels initial returned final) := by
  unfold forward at success
  split at success
  next rows cols logitsShape =>
    by_cases labelsShape : callerLabels.shape=[rows]
    · rw [if_pos labelsShape] at success
      by_cases zero : cols=0
      · rw [if_pos zero] at success
        cases success
      · rw [if_neg zero] at success
        obtain ⟨world,sized,hsize,success⟩ := bind_success _ _ _ _ success
        obtain ⟨rank,ranked,hrank,success⟩ := bind_success _ _ _ _ success
        obtain ⟨labels,labelsReady,hlabels,success⟩ := bind_success _ _ _ _ success
        obtain ⟨inplaceLogits,normalized,hnormalize,success⟩ := bind_success _ _ _ _ success
        by_cases overlap : (o.inplace && decide (rows>1 ∧ rowStride inplaceLogits<cols))=Bool.true
        · rw [if_pos overlap] at success
          cases success
        · rw [if_neg overlap] at success
          obtain ⟨resolvedLogits,resolvedState,hresolve,success⟩ := bind_success _ _ _ _ success
          obtain ⟨workingLogits,resolved,hworking,success⟩ := bind_success _ _ _ _ success
          obtain ⟨losses,lossAllocated,hloss,success⟩ := bind_success _ _ _ _ success
          obtain ⟨lse,lseAllocated,hlse,success⟩ := bind_success _ _ _ _ success
          obtain ⟨zlosses,zAllocated,hz,success⟩ := bind_success _ _ _ _ success
          obtain ⟨u,computed,hlaunch,success⟩ := bind_success _ _ _ _ success
          cases u
          obtain ⟨result,aggregated,haggregate,success⟩ := bind_success _ _ _ _ success
          obtain ⟨u,saved,hsave,success⟩ := bind_success _ _ _ _ success
          cases u
          obtain ⟨u,marked,hmark,success⟩ := bind_success _ _ _ _ success
          cases u
          simp only [Step.ok.injEq] at success
          obtain ⟨rfl,rfl⟩ := success
          exact ⟨⟨rows,cols,logitsShape,labelsShape,zero,world,sized,hsize,rank,ranked,hrank,
            labels,labelsReady,hlabels,inplaceLogits,normalized,hnormalize,overlap,
            resolvedLogits,resolvedState,hresolve,workingLogits,resolved,hworking,
            losses,lossAllocated,hloss,lse,lseAllocated,hlse,zlosses,zAllocated,hz,
            computed,hlaunch,result,aggregated,haggregate,saved,hsave,hmark,rfl⟩⟩
    · rw [if_neg labelsShape] at success
      cases success
  next other shape => cases success

structure BackwardData {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (c : Context) (callerUpstream output : Tensor) (initial final : World α σ) where
  upstream : Tensor
  upstreamReady : World α σ
  snapshotCall : CrossEntropy1.Host.snapshot rt.toRuntime callerUpstream initial=.ok upstream upstreamReady
  first : Tensor
  allocated : World α σ
  firstCall : (if c.options.inplace then .ok c.logits upstreamReady else
    rt.emptyLike c.logits Bool.false upstreamReady)=Step.ok first allocated
  outputIdentity : output=(if c.options.inplace then c.inplaceLogits else first)
  work : Tensor
  prepared : World α σ
  workCall : (if lastStride first≠1 then rt.emptyLike c.logits Bool.true allocated else
    .ok first allocated)=Step.ok work prepared
  rows : Nat
  cols : Nat
  shape : c.logits.shape=[rows,cols]
  nonzero : bwdWidth cols≠0
  computed : World α σ
  launchCall : backwardLaunch M rt c.logits.device (backwardParameters c work upstream rows cols (bwdWidth cols))
    (CrossEntropy1.Math.blocks cols (bwdWidth cols)) prepared=.ok () computed
  finish : (work.objectId=output.objectId ∧ final=computed) ∨
    (work.objectId≠output.objectId ∧ ∃ returned, rt.copy output work computed=.ok returned final)

theorem backward_success {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (c : Context) (callerUpstream output : Tensor) (zUpstream : Option Tensor) (initial final : World α σ)
    (success : backward M rt c callerUpstream zUpstream initial=.ok output final) :
    Nonempty (BackwardData M rt c callerUpstream output initial final) := by
  unfold backward at success
  obtain ⟨upstream,upstreamReady,hs,success⟩ := bind_success _ _ _ _ success
  obtain ⟨first,allocated,hf,success⟩ := bind_success _ _ _ _ success
  obtain ⟨work,prepared,hw,success⟩ := bind_success _ _ _ _ success
  split at success
  next rows cols shape =>
    dsimp only at success
    by_cases nonzero : bwdWidth cols=0
    · rw [if_pos nonzero] at success
      cases success
    · rw [if_neg nonzero] at success
      obtain ⟨u,computed,hl,success⟩ := bind_success _ _ _ _ success
      cases u
      by_cases different : work.objectId≠(if c.options.inplace then c.inplaceLogits else first).objectId
      · rw [if_pos different] at success
        obtain ⟨copied,closed,hc,success⟩ := bind_success _ _ _ _ success
        simp only [Step.ok.injEq] at success
        obtain ⟨rfl,rfl⟩ := success
        exact ⟨⟨upstream,upstreamReady,hs,first,allocated,hf,rfl,work,prepared,hw,
          rows,cols,shape,nonzero,computed,hl,Or.inr ⟨different,copied,hc⟩⟩⟩
      · rw [if_neg different] at success
        simp only [Step.ok.injEq] at success
        obtain ⟨rfl,rfl⟩ := success
        exact ⟨⟨upstream,upstreamReady,hs,first,allocated,hf,rfl,work,prepared,hw,
          rows,cols,shape,nonzero,computed,hl,Or.inl ⟨not_not.mp different,rfl⟩⟩⟩
  next other shape => cases success

#axiomsClean bind_success
#axiomsClean forward_launch_success
#axiomsClean backward_launch_success
#axiomsClean forward_success
#axiomsClean backward_success
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Host
