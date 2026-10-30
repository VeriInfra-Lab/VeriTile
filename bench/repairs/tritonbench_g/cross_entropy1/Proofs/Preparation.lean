import bench.repairs.tritonbench_g.cross_entropy1.Proofs.ForwardTrace
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Geometry

namespace VeriTile.Bench.Repairs.CrossEntropy1.Host
open VeriTile.Triton Source
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor SameMetadata)
noncomputable section
set_option maxHeartbeats 4000000

theorem Materialized.refl {α σ : Type} [Inhabited α] (obs : Observations α σ)
    (t : Tensor) (s : World α σ) (live : s.live t.region)
    (rank : t.strides.length=t.shape.length) : Materialized obs t t s s :=
  ⟨⟨rfl,rfl,rfl,rfl⟩,Preserves.refl obs s,live,fun _ _ _ => rfl,fun _ _ _ => rfl,rank⟩

theorem Materialized.trans {α σ : Type} [Inhabited α] {obs : Observations α σ}
    {a b c : Tensor} {s t u : World α σ}
    (ab : Materialized obs a b s t) (bc : Materialized obs b c t u) : Materialized obs a c s u := by
  refine ⟨⟨bc.metadata.shape.trans ab.metadata.shape,bc.metadata.dtype.trans ab.metadata.dtype,
    bc.metadata.cuda.trans ab.metadata.cuda,bc.metadata.device.trans ab.metadata.device⟩,
    ab.frame.trans bc.frame,bc.live,?_,?_,bc.rank⟩
  · intro dtype i hi
    exact (bc.floats (by simpa [ab.metadata.dtype] using dtype) i (by simpa [ab.metadata.numel] using hi)).trans
      (ab.floats dtype i hi)
  · intro dtype i hi
    exact (bc.integers (by simpa [ab.metadata.dtype] using dtype) i (by simpa [ab.metadata.numel] using hi)).trans
      (ab.integers dtype i hi)

theorem Materialized.prepend {α σ : Type} [Inhabited α] {obs : Observations α σ}
    {a b : Tensor} {s t u : World α σ} (frame : Preserves obs s t) (live : s.live a.region)
    (mat : Materialized obs a b t u) : Materialized obs a b s u := by
  refine ⟨mat.metadata,frame.trans mat.frame,mat.live,?_,?_,mat.rank⟩
  · intro dtype i hi
    exact (mat.floats dtype i hi).trans (frame.floatValue a live i)
  · intro dtype i hi
    exact (mat.integers dtype i hi).trans (frame.integers a i live)

theorem SnapshotValues.prepend {α σ : Type} [Inhabited α] {obs : Observations α σ}
    {a b : Tensor} {s t u : World α σ} (frame : Preserves obs s t) (live : s.live a.region)
    (snap : SnapshotValues obs a b t u) : SnapshotValues obs a b s u := by
  refine ⟨snap.materialized.prepend frame live,snap.dense,?_,?_⟩
  · intro old
    exact snap.fresh (frame.live _ old)
  · intro dtype i hi
    rw [snap.labels dtype i hi,frame.integers a i live]

theorem normalization_values {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (lib : Library obs rt) (input output : Tensor) (before after : World α σ)
    (live : before.live input.region) (rank : input.strides.length=input.shape.length)
    (success : (if lastStride input≠1 then rt.contiguous input before else .ok input before)=Step.ok output after) :
    Materialized obs input output before after ∧ RowLayout output ∧
      (input.isNeg=Bool.false → output.isNeg=Bool.false) := by
  by_cases needed : lastStride input≠1
  · rw [if_pos needed] at success
    exact lib.contiguous input before output after live success
  · rw [if_neg needed] at success
    cases success
    exact ⟨Materialized.refl obs input before live rank,last_stride_layout input rank (not_not.mp needed),id⟩

/-- Process-group metadata is an ordinary runtime contract. Group size/rank
are fixed properties of this group and both queries preserve tensor storage. -/
structure Groups {α σ : Type} (obs : Observations α σ) (rt : Runtime α σ) where
  size : Nat → Nat
  rank : Nat → Nat
  positive : ∀ group, 0<size group
  rankBound : ∀ group, rank group<size group
  sizeCall : ∀ group before result after, rt.worldSize group before=.ok result after →
    result=size group ∧ Preserves obs before after
  rankCall : ∀ group before result after, rt.rank group before=.ok result after →
    result=rank group ∧ Preserves obs before after

theorem forward_group_metadata {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (groups : Groups obs rt) {M : FP32 α} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    (e : ForwardData M rt o logits labels initial returned final) :
    0<e.world ∧ e.rank<e.world ∧ Preserves obs initial e.ranked := by
  have hs := e.sizeCall
  have hr := e.rankCall
  cases group : o.processGroup with
  | none =>
    simp only [groupSize,group,Step.ok.injEq] at hs
    simp only [groupRank,group,Step.ok.injEq] at hr
    have frame : Preserves obs initial e.ranked := by
      rw [← hr.2,← hs.2]
      exact Preserves.refl obs initial
    exact ⟨by omega,by omega,frame⟩
  | some g =>
    simp only [groupSize,group] at hs
    simp only [groupRank,group] at hr
    have sizes := groups.sizeCall _ _ _ _ hs
    have ranks := groups.rankCall _ _ _ _ hr
    exact ⟨by simpa [sizes.1] using groups.positive g,
      by simpa [sizes.1,ranks.1] using groups.rankBound g,sizes.2.trans ranks.2⟩

/-- Logical caller inputs are connected to the exact working tensor and
fresh label snapshot consumed by the actual source launch. Both normalization
stages preserve the saved original in-place object recorded in ForwardData. -/
theorem forward_preparation {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (lib : Library obs rt) (groups : Groups obs rt) {M : FP32 α} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    (e : ForwardData M rt o logits labels initial returned final)
    (logitsLive : initial.live logits.region) (labelsLive : initial.live labels.region)
    (rank : logits.strides.length=logits.shape.length) :
    SnapshotValues obs labels e.labels initial e.labelsReady ∧
    Materialized obs logits e.workingLogits initial e.resolved ∧ RowLayout e.workingLogits ∧
      e.workingLogits.isNeg=Bool.false ∧ Preserves obs e.labelsReady e.resolved := by
  have groupData := forward_group_metadata obs rt groups e
  have snapshot := snapshot_values obs rt lib labels e.labels e.ranked e.labelsReady
    (groupData.2.2.live _ labelsLive) e.snapshotCall
  have labelsBefore := snapshot.prepend groupData.2.2 labelsLive
  have normalized := normalization_values obs rt lib logits e.inplaceLogits e.labelsReady e.normalized
    (labelsBefore.materialized.frame.live _ logitsLive) rank e.originalNormalize
  have original := normalized.1.prepend labelsBefore.materialized.frame logitsLive
  have resolved := lib.resolve _ _ _ _ original.live e.resolveCall
  have working := normalization_values obs rt lib e.resolvedLogits e.workingLogits e.resolvedState e.resolved
    resolved.1.live resolved.1.rank e.workingNormalize
  exact ⟨labelsBefore,(original.trans resolved.1).trans working.1,working.2.1,working.2.2 resolved.2,normalized.1.frame.trans (resolved.1.frame.trans working.1.frame)⟩

#axiomsClean forward_preparation
#axiomsClean forward_group_metadata
#axiomsClean normalization_values
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Host
