import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Storage

namespace VeriTile.Bench.Repairs.CrossEntropy1.Physical
open VeriTile.Triton Source Host
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 6000000

/-- One ordinary runtime call, with its actual arguments and return. No source
kernel, complete wrapper, final-value predicate, or success oracle occurs here. -/
inductive Primitive {α σ : Type} (rt : Runtime α σ) : World α σ → World α σ → Prop
  | worldSize {g a n b} (call : rt.worldSize g a=.ok n b) : Primitive rt a b
  | rank {g a n b} (call : rt.rank g a=.ok n b) : Primitive rt a b
  | resolve {t a u b} (call : rt.resolveNeg t a=.ok u b) : Primitive rt a b
  | clone {t a u b} (call : rt.cloneContiguous t a=.ok u b) : Primitive rt a b
  | contiguous {t a u b} (call : rt.contiguous t a=.ok u b) : Primitive rt a b
  | empty {shape dtype device a t b} (call : rt.empty shape dtype device a=.ok t b) : Primitive rt a b
  | emptyLike {t packed a u b} (call : rt.emptyLike t packed a=.ok u b) : Primitive rt a b
  | enter {device a b} (call : rt.enterDevice device a=.ok () b) : Primitive rt a b
  | setupForward {p warps a b} (call : rt.setupForward p warps a=.ok () b) : Primitive rt a b
  | setupBackward {p warps a b} (call : rt.setupBackward p warps a=.ok () b) : Primitive rt a b
  | exit {device a b} (call : rt.exitDevice device a=.ok () b) : Primitive rt a b
  | lse {t a u b} (call : rt.logsumexp0 t a=.ok u b) : Primitive rt a b
  | sum {t a u b} (call : rt.sum0 t a=.ok u b) : Primitive rt a b
  | gather {dst src group a b} (call : rt.allGather dst src group a=.ok () b) : Primitive rt a b
  | reduce {t group a handle b} (call : rt.allReduceAsync t group a=.ok handle b) : Primitive rt a b
  | wait {handle a b} (call : rt.wait handle a=.ok () b) : Primitive rt a b
  | add {dst src a u b} (call : rt.addInplace dst src a=.ok u b) : Primitive rt a b
  | square {t a u b} (call : rt.square t a=.ok u b) : Primitive rt a b
  | scale {scale t a u b} (call : rt.scalarMul scale t a=.ok u b) : Primitive rt a b
  | equal {t target a u b} (call : rt.equalInt t target a=.ok u b) : Primitive rt a b
  | mask {dst mask a u b} (call : rt.maskedFillZero dst mask a=.ok u b) : Primitive rt a b
  | save {x l y original a b} (call : rt.saveTensors x l y original a=.ok () b) : Primitive rt a b
  | copy {dst src a u b} (call : rt.copy dst src a=.ok u b) : Primitive rt a b

/-- Python references held outside the invocation keep their allocations
active and their allocation descriptors stable. Dead internal temporaries are
not kept and can be retired or have their addresses reused. -/
structure Retains {σ : Type} (layout : Layout σ) (kept : RegionName → Prop) (a b : σ) : Prop where
  active : ∀ r, kept r → layout.active a r → layout.active b r
  buffer : ∀ r, kept r → layout.active a r → layout.buffer b r=layout.buffer a r

theorem Retains.refl {σ : Type} (layout : Layout σ) (kept : RegionName → Prop) (s : σ) :
    Retains layout kept s s := ⟨fun _ _ h => h,fun _ _ _ => rfl⟩

theorem Retains.trans {σ : Type} {layout : Layout σ} {kept : RegionName → Prop} {a b c : σ}
    (ab : Retains layout kept a b) (bc : Retains layout kept b c) : Retains layout kept a c :=
  ⟨fun r hp ha => bc.active r hp (ab.active r hp ha),
   fun r hp ha => (bc.buffer r hp (ab.active r hp ha)).trans (ab.buffer r hp ha)⟩

structure LifetimeLibrary {α σ : Type} (layout : Layout σ) (rt : Runtime α σ) (kept : RegionName → Prop) : Prop where
  call : ∀ a b, Primitive rt a b → Retains layout kept a.native b.native

theorem snapshot_lifetime {α σ : Type} (layout : Layout σ) (rt : Runtime α σ) (kept : RegionName → Prop)
    (lib : LifetimeLibrary layout rt kept) (input output : Tensor) (a b : World α σ)
    (call : snapshot rt input a=.ok output b) : Retains layout kept a.native b.native := by
  obtain ⟨trace⟩ := snapshot_success rt input output a b call
  exact (lib.call _ _ (.resolve trace.resolveCall)).trans (lib.call _ _ (.clone trace.cloneCall))

theorem normalization_lifetime {α σ : Type} (layout : Layout σ) (rt : Runtime α σ) (kept : RegionName → Prop)
    (lib : LifetimeLibrary layout rt kept) (input output : Tensor) (a b : World α σ)
    (call : (if lastStride input≠1 then rt.contiguous input a else .ok input a)=Step.ok output b) :
    Retains layout kept a.native b.native := by
  by_cases needed : lastStride input≠1
  · exact lib.call _ _ (.contiguous (by simpa only [if_pos needed] using call))
  · simp only [if_neg needed,Step.ok.injEq] at call
    rw [← call.2]
    exact Retains.refl _ _ _

theorem launch_lifetime {α σ : Type} (layout : Layout σ) (rt : Runtime α σ) (kept : RegionName → Prop)
    (lib : LifetimeLibrary layout rt kept) (setup : World α σ → Step (World α σ) Unit)
    (setupLocal : ∀ a b, setup a=.ok () b → Primitive rt a b)
    (kernel : Memory α → Memory α) (device : Nat) (a b : World α σ)
    (trace : LaunchData rt setup kernel device a b) : Retains layout kept a.native b.native := by
  have first := (lib.call _ _ (.enter trace.enterCall)).trans (lib.call _ _ (setupLocal _ _ trace.setupCall))
  have last := lib.call _ _ (.exit trace.exitCall)
  exact first.trans last

theorem local_lifetime {α σ : Type} (layout : Layout σ) (rt : Runtime α σ) (kept : RegionName → Prop)
    (lib : LifetimeLibrary layout rt kept) {count : Nat} {losses lse : Tensor} {a b : World α σ} {pair : Tensor × Tensor}
    (trace : LocalTrace rt count losses lse a pair b) : Retains layout kept a.native b.native := by
  cases trace with
  | direct => exact Retains.refl _ _ _
  | reduced _ _ _ _ _ lc sc => exact (lib.call _ _ (.lse lc)).trans (lib.call _ _ (.sum sc))

theorem rank_lifetime {α σ : Type} (layout : Layout σ) (rt : Runtime α σ) (kept : RegionName → Prop)
    (lib : LifetimeLibrary layout rt kept) {o : Options} {world rows : Nat} {input pair : Tensor × Tensor} {a b : World α σ}
    (trace : RankTrace rt o world rows input a pair b) : Retains layout kept a.native b.native := by
  cases trace with
  | direct => exact Retains.refl _ _ _
  | reduced _ _ _ _ _ _ _ _ _ ac gc sc lc wc =>
    exact ((((lib.call _ _ (.empty ac)).trans (lib.call _ _ (.gather gc))).trans
      (lib.call _ _ (.reduce sc))).trans (lib.call _ _ (.lse lc))).trans (lib.call _ _ (.wait wc))

theorem correction_lifetime {α σ : Type} (layout : Layout σ) (rt : Runtime α σ) (kept : RegionName → Prop)
    (lib : LifetimeLibrary layout rt kept) {scale : ℝ} {losses lse output : Tensor} {a b : World α σ}
    (trace : CorrectionTrace rt scale losses lse a output b) : Retains layout kept a.native b.native := by
  cases trace with
  | zero => exact Retains.refl _ _ _
  | nonzero _ _ _ _ _ _ _ sc mc ac =>
    exact ((lib.call _ _ (.square sc)).trans (lib.call _ _ (.scale mc))).trans (lib.call _ _ (.add ac))

theorem aggregation_lifetime {α σ : Type} (layout : Layout σ) (rt : Runtime α σ) (kept : RegionName → Prop)
    (lib : LifetimeLibrary layout rt kept) {o : Options} {world rows count : Nat}
    {labels losses lse : Tensor} {a b : World α σ} {returned : Tensor × Tensor}
    (trace : AggregateData rt o world rows count labels losses lse a returned b) : Retains layout kept a.native b.native := by
  have locals := local_lifetime layout rt kept lib trace.localCalls
  have ranks := rank_lifetime layout rt kept lib trace.rankCalls
  have correction := correction_lifetime layout rt kept lib trace.correctionCalls
  exact ((((locals.trans ranks).trans (lib.call _ _ (.add trace.addCall))).trans correction).trans
    (lib.call _ _ (.equal trace.compareCall))).trans (lib.call _ _ (.mask trace.maskCall))

theorem forward_lifetime {α σ : Type} [Inhabited α] (layout : Layout σ) (rt : Runtime α σ) (kept : RegionName → Prop)
    (lib : LifetimeLibrary layout rt kept) {M : FP32 α} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    (f : ForwardData M rt o logits labels initial returned final) :
    Retains layout kept initial.native final.native := by
  have groups : Retains layout kept initial.native f.ranked.native := by
    have hs := f.sizeCall
    have hr := f.rankCall
    cases group : o.processGroup with
    | none =>
      simp only [groupSize,group,Step.ok.injEq] at hs
      simp only [groupRank,group,Step.ok.injEq] at hr
      rw [← hr.2,← hs.2]
      exact Retains.refl _ _ _
    | some g =>
      exact (lib.call _ _ (.worldSize (by simpa only [groupSize,group] using hs))).trans
        (lib.call _ _ (.rank (by simpa only [groupRank,group] using hr)))
  have snap := snapshot_lifetime layout rt kept lib _ _ _ _ f.snapshotCall
  have original := normalization_lifetime layout rt kept lib _ _ _ _ f.originalNormalize
  have resolve := lib.call _ _ (.resolve f.resolveCall)
  have work := normalization_lifetime layout rt kept lib _ _ _ _ f.workingNormalize
  have losses := lib.call _ _ (.empty f.lossCall)
  have lses := lib.call _ _ (.empty f.lseCall)
  obtain ⟨launch⟩ := launch_success rt (rt.setupForward f.parameters (warps f.parameters.block))
    (fun m => Grid.forward M f.parameters m f.count) f.workingLogits.device f.lseAllocated f.computed f.launchCall
  have kernel := launch_lifetime layout rt kept lib _ (fun _ _ call => .setupForward call) _ _ _ _ launch
  have aggregate : Retains layout kept f.computed.native f.aggregated.native := by
    obtain ⟨aggregation⟩ := forward_aggregation f
    cases aggregation with
    | direct _ _ state => rw [state]; exact Retains.refl _ _ _
    | split _ trace => exact aggregation_lifetime layout rt kept lib trace
  exact ((((((((groups.trans snap).trans original).trans resolve).trans work).trans losses).trans lses).trans kernel).trans
    aggregate).trans (lib.call _ _ (.save f.saveCall))

theorem backward_lifetime {α σ : Type} [Inhabited α] (layout : Layout σ) (rt : Runtime α σ) (kept : RegionName → Prop)
    (lib : LifetimeLibrary layout rt kept) {M : FP32 α} {c : Context} {upstream output : Tensor}
    {initial final : World α σ} (b : BackwardData M rt c upstream output initial final) :
    Retains layout kept initial.native final.native := by
  have snap := snapshot_lifetime layout rt kept lib _ _ _ _ b.snapshotCall
  have first : Retains layout kept b.upstreamReady.native b.allocated.native := by
    have call := b.firstCall
    by_cases inplace : c.options.inplace=Bool.true
    · simp only [if_pos inplace,Step.ok.injEq] at call
      rw [← call.2]
      exact Retains.refl _ _ _
    · exact lib.call _ _ (.emptyLike (by simpa only [if_neg inplace] using call))
  have work : Retains layout kept b.allocated.native b.prepared.native := by
    have call := b.workCall
    by_cases packed : lastStride b.first≠1
    · exact lib.call _ _ (.emptyLike (by simpa only [if_pos packed] using call))
    · simp only [if_neg packed,Step.ok.injEq] at call
      rw [← call.2]
      exact Retains.refl _ _ _
  obtain ⟨launch⟩ := launch_success rt (rt.setupBackward b.parameters (warps b.parameters.block))
    (fun m => Grid.backward M b.parameters m b.count) c.logits.device b.prepared b.computed b.launchCall
  have kernel := launch_lifetime layout rt kept lib _ (fun _ _ call => .setupBackward call) _ _ _ _ launch
  have finish : Retains layout kept b.computed.native final.native := by
    rcases b.finish with ⟨same,state⟩|⟨different,returned,call⟩
    · simpa only [state] using Retains.refl layout kept b.computed.native
    · exact lib.call _ _ (.copy call)
  exact (((snap.trans first).trans work).trans kernel).trans finish

#axiomsClean snapshot_lifetime
#axiomsClean aggregation_lifetime
#axiomsClean forward_lifetime
#axiomsClean backward_lifetime
#specNonCircular Primitive avoiding [Source.forward,Source.backward,Grid.forward,Grid.backward,Host.forward,Host.backward]
#specNonCircular LifetimeLibrary avoiding [Source.forward,Source.backward,Grid.forward,Grid.backward,Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Physical
