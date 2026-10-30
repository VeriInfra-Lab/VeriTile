import bench.tritonbench_g.cross_entropy1.Proofs.Host

namespace VeriTile.Bench.Repairs.CrossEntropy1.Host
open VeriTile.Triton Source
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

structure DeviceTrace {α σ A : Type} (rt : Runtime α σ) (device : Nat)
    (body : World α σ → Step (World α σ) A) (initial : World α σ) (value : A) (final : World α σ) where
  entered : World α σ
  computed : World α σ
  enterCall : rt.enterDevice device initial=.ok () entered
  bodyCall : body entered=.ok value computed
  exitCall : rt.exitDevice device computed=.ok () final

theorem withDevice_success {α σ A : Type} (rt : Runtime α σ) (device : Nat)
    (body : World α σ → Step (World α σ) A) (initial : World α σ) (value : A) (final : World α σ)
    (success : withDevice rt device body initial=.ok value final) :
    Nonempty (DeviceTrace rt device body initial value final) := by
  cases he : rt.enterDevice device initial with
  | error e s => simp [withDevice,bind,he] at success
  | ok u entered =>
    cases u
    cases hb : body entered with
    | error e failed =>
      cases hx : rt.exitDevice device failed <;> simp [withDevice,bind,he,hb,hx] at success
    | ok a computed =>
      cases hx : rt.exitDevice device computed with
      | error e failed => simp [withDevice,bind,he,hb,hx] at success
      | ok u closed =>
        cases u
        simp only [withDevice,bind,he,hb,hx,Step.ok.injEq] at success
        obtain ⟨rfl,rfl⟩ := success
        exact ⟨⟨entered,computed,he,hb,hx⟩⟩

structure LaunchData {α σ : Type} (rt : Runtime α σ)
    (setup : World α σ → Step (World α σ) Unit) (kernel : Memory α → Memory α)
    (device : Nat) (initial final : World α σ) where
  entered : World α σ
  ready : World α σ
  enterCall : rt.enterDevice device initial=.ok () entered
  setupCall : setup entered=.ok () ready
  exitCall : rt.exitDevice device {ready with memory:=kernel ready.memory}=.ok () final

theorem launch_success {α σ : Type} (rt : Runtime α σ)
    (setup : World α σ → Step (World α σ) Unit) (kernel : Memory α → Memory α)
    (device : Nat) (initial final : World α σ)
    (success : withDevice rt device (fun entered => bind (setup entered) fun _ ready =>
      .ok () {ready with memory:=kernel ready.memory}) initial=.ok () final) :
    Nonempty (LaunchData rt setup kernel device initial final) := by
  obtain ⟨e⟩ := withDevice_success rt device _ initial () final success
  cases hs : setup e.entered with
  | error reason failed =>
    have h := e.bodyCall
    simp [bind,hs] at h
  | ok u ready =>
    cases u
    have eq : {ready with memory:=kernel ready.memory}=e.computed := by
      have h := e.bodyCall
      simpa only [bind,hs,Step.ok.injEq,true_and] using h
    refine ⟨⟨e.entered,ready,e.enterCall,hs,?_⟩⟩
    simpa only [eq] using e.exitCall

structure SnapshotData {α σ : Type} (rt : Runtime α σ) (input output : Tensor)
    (initial final : World α σ) where
  resolved : Tensor
  intermediate : World α σ
  resolveCall : rt.resolveNeg input initial=.ok resolved intermediate
  cloneCall : rt.cloneContiguous resolved intermediate=.ok output final

theorem snapshot_success {α σ : Type} (rt : Runtime α σ) (input output : Tensor)
    (initial final : World α σ) (success : snapshot rt input initial=.ok output final) :
    Nonempty (SnapshotData rt input output initial final) := by
  cases hr : rt.resolveNeg input initial with
  | error reason failed => simp [snapshot,bind,hr] at success
  | ok resolved intermediate =>
    exact ⟨⟨resolved,intermediate,hr,by simpa only [snapshot,bind,hr] using success⟩⟩

/-- Trace of actual backward calls and return identity. Its fields do not
assume a gradient formula or a successful kernel numerical result. -/
structure BackwardData {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (c : Context) (callerUpstream output : Tensor) (initial final : World α σ) where
  upstream : Tensor
  upstreamReady : World α σ
  snapshotCall : snapshot rt callerUpstream initial=.ok upstream upstreamReady
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
    (Math.blocks cols (bwdWidth cols)) prepared=.ok () computed
  finish : (work.objectId=output.objectId ∧ final=computed) ∨
    (work.objectId≠output.objectId ∧ ∃ returned,
      rt.copy output work computed=.ok returned final)

theorem backward_success {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (c : Context) (callerUpstream output : Tensor) (initial final : World α σ)
    (success : backward M rt c callerUpstream initial=.ok output final) :
    Nonempty (BackwardData M rt c callerUpstream output initial final) := by
  unfold backward at success
  cases hs : snapshot rt callerUpstream initial with
  | error reason failed => simp [bind,hs] at success
  | ok upstream upstreamReady =>
    simp only [bind,hs] at success
    cases hf : (if c.options.inplace then .ok c.logits upstreamReady else rt.emptyLike c.logits Bool.false upstreamReady) with
    | error reason failed => simp [bind,hf] at success
    | ok first allocated =>
      simp only [bind,hf] at success
      cases hw : (if lastStride first≠1 then rt.emptyLike c.logits Bool.true allocated else .ok first allocated) with
      | error reason failed => simp [bind,hw] at success
      | ok work prepared =>
        simp only [bind,hw] at success
        split at success
        next rows cols shape =>
          split at success
          next zero => cases success
          next nonzero =>
            cases hl : backwardLaunch M rt c.logits.device
                (backwardParameters c work upstream rows cols (bwdWidth cols))
                (Math.blocks cols (bwdWidth cols)) prepared with
            | error reason failed => simp [bind,hl] at success
            | ok u computed =>
              cases u
              simp only [bind,hl] at success
              by_cases different : work.objectId≠(if c.options.inplace then c.inplaceLogits else first).objectId
              · rw [if_pos different] at success
                cases hc : rt.copy (if c.options.inplace then c.inplaceLogits else first) work computed with
                | error reason failed => simp [bind,hc] at success
                | ok copied closed =>
                  simp only [bind,hc,Step.ok.injEq] at success
                  obtain ⟨rfl,rfl⟩ := success
                  exact ⟨⟨upstream,upstreamReady,hs,first,allocated,hf,rfl,work,prepared,hw,
                    rows,cols,shape,nonzero,computed,hl,Or.inr ⟨different,copied,hc⟩⟩⟩
              · rw [if_neg different] at success
                simp only [Step.ok.injEq] at success
                obtain ⟨rfl,rfl⟩ := success
                exact ⟨⟨upstream,upstreamReady,hs,first,allocated,hf,rfl,work,prepared,hw,
                  rows,cols,shape,nonzero,computed,hl,Or.inl ⟨not_not.mp different,rfl⟩⟩⟩
        next other shape => cases success

#axiomsClean withDevice_success
#axiomsClean launch_success
#axiomsClean snapshot_success
#axiomsClean backward_success
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Host
