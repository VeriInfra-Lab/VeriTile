import bench.repairs.tritonbench_g.cross_entropy1.Proofs.DenseGeometry
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.TensorOps

namespace VeriTile.Bench.Repairs.CrossEntropy1.Host
open VeriTile.Triton Source
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 6000000

/-- Intermediate saved-view geometry. The final forward/backward composition
must derive these fields from the actual forward trace and ordinary view
contracts; they are not a substitute for that linkage. -/
structure BackwardContext {α σ : Type} (rt : Runtime α σ) (objects : ObjectRegistry rt)
    (c : Context) (initial : World α σ) : Prop where
  logitsLive : initial.live c.logits.region
  labelsLive : initial.live c.labels.region
  lseLive : initial.live c.lse.region
  destinationLive : initial.live c.inplaceLogits.region
  logitsRank : c.logits.strides.length=c.logits.shape.length
  logitsNotNeg : c.logits.isNeg=Bool.false
  logitsObject : objects.valid c.logits
  destinationObject : objects.valid c.inplaceLogits
  destinationShape : c.inplaceLogits.shape=c.logits.shape
  logitsLabels : c.logits.region≠c.labels.region
  logitsLSE : c.logits.region≠c.lse.region
  rows : c.options.inplace=Bool.true → ∀ rows cols, c.logits.shape=[rows,cols] → rows≤1 ∨ cols≤rowStride c.logits

structure BufferInfo {α σ : Type} (obs : Observations α σ) (rt : Runtime α σ) (objects : ObjectRegistry rt)
    (c : Context) (upstream : Tensor) (rows cols : Nat) (before after : World α σ) (buffer : Tensor) : Prop where
  shape : buffer.shape=c.logits.shape
  rank : buffer.strides.length=buffer.shape.length
  notNeg : buffer.isNeg=Bool.false
  registered : objects.valid buffer
  frame : Preserves obs before after
  live : after.live buffer.region
  labels : buffer.region≠c.labels.region
  lse : buffer.region≠c.lse.region
  upstream : buffer.region≠upstream.region
  logitsAlias : buffer.region=c.logits.region → buffer=c.logits
  rows : lastStride buffer=1 → rows≤1 ∨ cols≤rowStride buffer

theorem allocated_buffer {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (lib : Library obs rt) (objects : ObjectRegistry rt) (c : Context) (upstream : Tensor)
    (rows cols : Nat) (shape : c.logits.shape=[rows,cols]) (packed : Bool) (before after : World α σ) (buffer : Tensor)
    (logitsLive : before.live c.logits.region) (labelsLive : before.live c.labels.region)
    (lseLive : before.live c.lse.region) (upstreamLive : before.live upstream.region)
    (call : rt.emptyLike c.logits packed before=.ok buffer after) :
    BufferInfo obs rt objects c upstream rows cols before after buffer ∧ (packed=Bool.true → buffer.Dense) := by
  have a := lib.emptyLike _ _ _ _ _ logitsLive call
  refine ⟨⟨a.1.shapeEq,a.1.rank,a.1.notNeg,objects.emptyLike _ _ _ _ _ call,a.1.frame,a.1.live,
    fresh_ne_live obs _ _ _ _ _ _ a.1 _ labelsLive,
    fresh_ne_live obs _ _ _ _ _ _ a.1 _ lseLive,
    fresh_ne_live obs _ _ _ _ _ _ a.1 _ upstreamLive,?_,?_⟩,a.2.1⟩
  · intro same
    exact False.elim ((fresh_ne_live obs _ _ _ _ _ _ a.1 _ logitsLive) same)
  · intro last
    exact a.2.2 rows cols (a.1.shapeEq.trans shape) last

theorem first_buffer {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (lib : Library obs rt) (objects : ObjectRegistry rt) {M : FP32 α} {c : Context}
    {callerUpstream output : Tensor} {initial final : World α σ}
    (e : BackwardData M rt c callerUpstream output initial final) (ctx : BackwardContext rt objects c initial)
    (snapshot : SnapshotValues obs callerUpstream e.upstream initial e.upstreamReady) :
    BufferInfo obs rt objects c e.upstream e.rows e.cols e.upstreamReady e.allocated e.first := by
  by_cases inplace : c.options.inplace=Bool.true
  · have first := e.firstCall
    simp only [if_pos inplace,Step.ok.injEq] at first
    have tensorEq : e.first=c.logits := first.1.symm
    have stateEq : e.allocated=e.upstreamReady := first.2.symm
    have upstreamSeparate : c.logits.region≠e.upstream.region := by
      intro same
      exact snapshot.fresh (same ▸ ctx.logitsLive)
    refine ⟨by simp only [tensorEq],by simpa only [tensorEq] using ctx.logitsRank,
      by simpa only [tensorEq] using ctx.logitsNotNeg,by simpa only [tensorEq] using ctx.logitsObject,
      ?_,?_,by simpa only [tensorEq] using ctx.logitsLabels,by simpa only [tensorEq] using ctx.logitsLSE,
      by simpa only [tensorEq] using upstreamSeparate,fun _ => tensorEq,?_⟩
    · simpa only [stateEq] using Preserves.refl obs e.upstreamReady
    · simpa only [tensorEq,stateEq] using snapshot.materialized.frame.live _ ctx.logitsLive
    · intro last
      simpa only [tensorEq] using ctx.rows inplace e.rows e.cols e.shape
  · have call : rt.emptyLike c.logits Bool.false e.upstreamReady=.ok e.first e.allocated := by
      simpa only [if_neg inplace] using e.firstCall
    exact (allocated_buffer obs rt lib objects c e.upstream e.rows e.cols e.shape Bool.false e.upstreamReady e.allocated e.first
      (snapshot.materialized.frame.live _ ctx.logitsLive) (snapshot.materialized.frame.live _ ctx.labelsLive)
      (snapshot.materialized.frame.live _ ctx.lseLive) snapshot.materialized.live call).1

structure WorkInfo {α σ : Type} (obs : Observations α σ) (rt : Runtime α σ) (objects : ObjectRegistry rt)
    (c : Context) (upstream : Tensor) (rows cols : Nat) (before after : World α σ) (tensor : Tensor) : Prop where
  buffer : BufferInfo obs rt objects c upstream rows cols before after tensor
  layout : RowLayout tensor
  rows : rows≤1 ∨ cols≤rowStride tensor

theorem work_buffer {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (lib : Library obs rt) (objects : ObjectRegistry rt) {M : FP32 α} {c : Context}
    {callerUpstream output : Tensor} {initial final : World α σ}
    (e : BackwardData M rt c callerUpstream output initial final) (ctx : BackwardContext rt objects c initial)
    (snapshot : SnapshotValues obs callerUpstream e.upstream initial e.upstreamReady)
    (first : BufferInfo obs rt objects c e.upstream e.rows e.cols e.upstreamReady e.allocated e.first) :
    WorkInfo obs rt objects c e.upstream e.rows e.cols e.allocated e.prepared e.work := by
  by_cases separate : lastStride e.first≠1
  · have call : rt.emptyLike c.logits Bool.true e.allocated=.ok e.work e.prepared := by
      simpa only [if_pos separate] using e.workCall
    have preparedFrame := snapshot.materialized.frame.trans first.frame
    have a := allocated_buffer obs rt lib objects c e.upstream e.rows e.cols e.shape Bool.true e.allocated e.prepared e.work
      (preparedFrame.live _ ctx.logitsLive) (preparedFrame.live _ ctx.labelsLive) (preparedFrame.live _ ctx.lseLive)
      (first.frame.live _ snapshot.materialized.live) call
    have dense := a.2 rfl
    exact ⟨a.1,dense_row_layout e.work dense a.1.rank,dense_rows e.work e.rows e.cols dense (a.1.shape.trans e.shape) a.1.rank⟩
  · have work := e.workCall
    simp only [if_neg separate,Step.ok.injEq] at work
    have tensorEq : e.work=e.first := work.1.symm
    have stateEq : e.prepared=e.allocated := work.2.symm
    have last : lastStride e.first=1 := not_not.mp separate
    refine ⟨⟨by simpa only [tensorEq] using first.shape,by simpa only [tensorEq] using first.rank,
      by simpa only [tensorEq] using first.notNeg,by simpa only [tensorEq] using first.registered,
      ?_,by simpa only [tensorEq,stateEq] using first.live,by simpa only [tensorEq] using first.labels,
      by simpa only [tensorEq] using first.lse,by simpa only [tensorEq] using first.upstream,
      ?_,by simpa only [tensorEq] using first.rows⟩,?_,?_⟩
    · simpa only [stateEq] using Preserves.refl obs e.allocated
    · intro same
      exact tensorEq.trans (first.logitsAlias (by simpa only [tensorEq] using same))
    · simpa only [tensorEq] using last_stride_layout e.first first.rank last
    · simpa only [tensorEq] using first.rows last

#axiomsClean first_buffer
#axiomsClean work_buffer
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Host
