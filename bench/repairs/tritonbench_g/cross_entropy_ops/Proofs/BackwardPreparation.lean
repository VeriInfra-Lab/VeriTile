import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.Preparation
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.BackwardKernel

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Host
open VeriTile.Triton
open CrossEntropy1.Source (FP32)
open CrossEntropy1.Host (World Step Observations SnapshotValues Preserves lastStride rowStride
  dense_row_layout dense_rows last_stride_layout)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

/-- Actual Ops backward trace composition; generic allocation, layout, object
and buffer contracts are unchanged CE1 helpers. No CE1 launch is synthesized. -/
theorem first_buffer {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (lib : LocalAPI obs rt) (objects : CrossEntropy1.Host.ObjectRegistry rt.toRuntime) {M : FP32 α} {c : Context}
    {callerUpstream output : Tensor} {initial final : World α σ}
    (e : BackwardData M rt c callerUpstream output initial final) (ctx : CrossEntropy1.Host.BackwardContext rt.toRuntime objects c.base initial)
    (snapshot : SnapshotValues obs callerUpstream e.upstream initial e.upstreamReady) :
    CrossEntropy1.Host.BufferInfo obs rt.toRuntime objects c.base e.upstream e.rows e.cols e.upstreamReady e.allocated e.first := by
  by_cases inplace : c.options.inplace=Bool.true
  · have first := e.firstCall
    simp only [if_pos inplace,Step.ok.injEq] at first
    have tensorEq : e.first=c.logits := first.1.symm
    have stateEq : e.allocated=e.upstreamReady := first.2.symm
    have upstreamSeparate : c.logits.region≠e.upstream.region := by
      intro same
      exact snapshot.fresh (same ▸ ctx.logitsLive)
    refine ⟨by simp only [tensorEq,Context.base],by simpa only [tensorEq] using ctx.logitsRank,
      by simpa only [tensorEq] using ctx.logitsNotNeg,by simpa only [tensorEq] using ctx.logitsObject,
      ?_,?_,by simpa only [tensorEq] using ctx.logitsLabels,by simpa only [tensorEq] using ctx.logitsLSE,
      by simpa only [tensorEq] using upstreamSeparate,fun _ => tensorEq,?_⟩
    · simpa only [stateEq] using Preserves.refl obs e.upstreamReady
    · simpa only [tensorEq,stateEq] using snapshot.materialized.frame.live _ ctx.logitsLive
    · intro last
      simpa only [tensorEq] using ctx.rows inplace e.rows e.cols e.shape
  · have call : rt.emptyLike c.logits Bool.false e.upstreamReady=.ok e.first e.allocated := by
      simpa only [if_neg inplace] using e.firstCall
    exact (CrossEntropy1.Host.allocated_buffer obs rt.toRuntime lib.tensor objects c.base e.upstream e.rows e.cols e.shape Bool.false e.upstreamReady e.allocated e.first
      (snapshot.materialized.frame.live _ ctx.logitsLive) (snapshot.materialized.frame.live _ ctx.labelsLive)
      (snapshot.materialized.frame.live _ ctx.lseLive) snapshot.materialized.live call).1

theorem work_buffer {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (lib : LocalAPI obs rt) (objects : CrossEntropy1.Host.ObjectRegistry rt.toRuntime) {M : FP32 α} {c : Context}
    {callerUpstream output : Tensor} {initial final : World α σ}
    (e : BackwardData M rt c callerUpstream output initial final) (ctx : CrossEntropy1.Host.BackwardContext rt.toRuntime objects c.base initial)
    (snapshot : SnapshotValues obs callerUpstream e.upstream initial e.upstreamReady)
    (first : CrossEntropy1.Host.BufferInfo obs rt.toRuntime objects c.base e.upstream e.rows e.cols e.upstreamReady e.allocated e.first) :
    CrossEntropy1.Host.WorkInfo obs rt.toRuntime objects c.base e.upstream e.rows e.cols e.allocated e.prepared e.work := by
  by_cases separate : lastStride e.first≠1
  · have call : rt.emptyLike c.logits Bool.true e.allocated=.ok e.work e.prepared := by
      simpa only [if_pos separate] using e.workCall
    have preparedFrame := snapshot.materialized.frame.trans first.frame
    have a := CrossEntropy1.Host.allocated_buffer obs rt.toRuntime lib.tensor objects c.base e.upstream e.rows e.cols e.shape Bool.true e.allocated e.prepared e.work
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
end VeriTile.Bench.Repairs.CrossEntropyOps.Host
