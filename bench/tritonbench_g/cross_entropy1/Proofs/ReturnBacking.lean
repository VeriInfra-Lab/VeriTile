import bench.tritonbench_g.cross_entropy1.Proofs.Lifetimes
import bench.tritonbench_g.cross_entropy1.Proofs.Descriptors

namespace VeriTile.Bench.Repairs.CrossEntropy1.Physical
open VeriTile.Triton Source Host
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 6000000

structure Backing {σ : Type} (layout : Layout σ) (native : σ) (t : Tensor) (width : Nat) : Prop where
  active : layout.active native t.region
  coordinates : ∀ i, i<t.numel → width*(t.index i+1)≤(layout.buffer native t.region).capacity

theorem Backing.retain {σ : Type} {layout : Layout σ} {kept : RegionName → Prop} {a b : σ}
    {t : Tensor} {width : Nat} (backed : Backing layout a t width) (life : Retains layout kept a b)
    (held : kept t.region) : Backing layout b t width :=
  ⟨life.active _ held backed.active,fun i hi => by rw [life.buffer _ held backed.active]; exact backed.coordinates i hi⟩

/-- Local tensor-object and backing guarantees. These validate actual method
operands/results, not arbitrary manufactured tensor records. No source
address calculation, cross-entropy value, or whole-wrapper result occurs. -/
structure BackingLibrary {α σ : Type} (layout : Layout σ) (rt : Runtime α σ) : Prop where
  empty : ∀ shape device before output after,
    rt.empty shape "torch.float32" device before=.ok output after → Backing layout after.native output 4
  emptyLike : ∀ input packed before output after, input.dtype="torch.float32" →
    rt.emptyLike input packed before=.ok output after → Backing layout after.native output 4
  mask : ∀ dst mask before output after, output.dtype="torch.float32" →
    rt.maskedFillZero dst mask before=.ok output after → Backing layout after.native output 4
  copy : ∀ dst src before output after, dst.dtype="torch.float32" →
    rt.copy dst src before=.ok output after → Backing layout after.native dst 4
  savedFloat : ∀ x l y original before after t, t=x ∨ t=l ∨ t=y ∨ t=original → t.dtype="torch.float32" →
    rt.saveTensors x l y original before=.ok () after → Backing layout after.native t 4
  savedInt : ∀ x l y original before after, y.dtype="torch.int64" →
    rt.saveTensors x l y original before=.ok () after → Backing layout after.native y 8

/-- Backing of the actual returned forward loss is obtained from its actual
allocation or final mask call, then retained through save-for-backward. -/
theorem forward_return_backing {α σ : Type} [Inhabited α] (layout : Layout σ) (rt : Runtime α σ)
    (kept : RegionName → Prop) (life : LifetimeLibrary layout rt kept) (physical : BackingLibrary layout rt)
    {obs : Observations α σ} {M : FP32 α} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    {f : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs f)
    (held : kept returned.losses.region) (dtype : returned.losses.dtype="torch.float32") :
    Backing layout final.native returned.losses 4 := by
  have identity : returned.losses=f.pair.1 := congrArg ForwardReturn.losses f.returnIdentity
  have pairHeld : kept f.pair.1.region := by simpa only [identity] using held
  have backing : Backing layout f.aggregated.native f.pair.1 4 := by
    obtain ⟨aggregation⟩ := forward_aggregation f
    cases aggregation with
    | direct flag pair state =>
      have loss := physical.empty _ _ _ _ _ f.lossCall
      have lifetime := (life.call _ _ (.empty f.lseCall)).trans
        (launch_lifetime layout rt kept life _ (fun _ _ call => .setupForward call) _ _ _ _ k.launch)
      have keptLoss : kept f.losses.region := by simpa only [pair] using pairHeld
      simpa only [pair,state] using loss.retain lifetime keptLoss
    | split flag trace =>
      exact physical.mask _ _ _ _ _ (by simpa only [identity] using dtype) trace.maskCall
  have saved := backing.retain (life.call _ _ (.save f.saveCall)) pairHeld
  simpa only [identity] using saved

theorem forward_saved_backing {α σ : Type} [Inhabited α] (layout : Layout σ) (rt : Runtime α σ)
    (physical : BackingLibrary layout rt) {M : FP32 α} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    (f : ForwardData M rt o logits labels initial returned final)
    (logitsDtype : returned.context.logits.dtype="torch.float32") (lseDtype : returned.context.lse.dtype="torch.float32")
    (labelsDtype : returned.context.labels.dtype="torch.int64") (destinationDtype : returned.context.inplaceLogits.dtype="torch.float32") :
    Backing layout final.native returned.context.logits 4 ∧ Backing layout final.native returned.context.lse 4 ∧
    Backing layout final.native returned.context.labels 8 ∧ Backing layout final.native returned.context.inplaceLogits 4 := by
  have logitsEq : returned.context.logits=f.workingLogits := congrArg (fun r : ForwardReturn => r.context.logits) f.returnIdentity
  have lseEq : returned.context.lse=f.pair.2 := congrArg (fun r : ForwardReturn => r.context.lse) f.returnIdentity
  have labelsEq : returned.context.labels=f.labels := congrArg (fun r : ForwardReturn => r.context.labels) f.returnIdentity
  have originalEq : returned.context.inplaceLogits=f.inplaceLogits := congrArg (fun r : ForwardReturn => r.context.inplaceLogits) f.returnIdentity
  exact ⟨physical.savedFloat _ _ _ _ _ _ _ (Or.inl logitsEq) logitsDtype f.saveCall,
    physical.savedFloat _ _ _ _ _ _ _ (Or.inr (Or.inl lseEq)) lseDtype f.saveCall,
    by simpa only [labelsEq] using physical.savedInt _ _ _ _ _ _ (by simpa only [labelsEq] using labelsDtype) f.saveCall,
    physical.savedFloat _ _ _ _ _ _ _ (Or.inr (Or.inr (Or.inr originalEq))) destinationDtype f.saveCall⟩

/-- Work backing follows the actual first allocation and packed-work branch.
The initial saved-logit backing is supplied by the preceding forward save. -/
theorem backward_work_backing {α σ : Type} [Inhabited α] (layout : Layout σ) (rt : Runtime α σ)
    (kept : RegionName → Prop) (life : LifetimeLibrary layout rt kept) (physical : BackingLibrary layout rt)
    {obs : Observations α σ} {objects : ObjectRegistry rt} {M : FP32 α} {c : Context}
    {callerUpstream output : Tensor} {initial final : World α σ}
    {b : BackwardData M rt c callerUpstream output initial final} (k : BackwardKernelData obs objects b)
    (heldLogits : kept c.logits.region) (heldWork : kept b.work.region) (dtype : c.logits.dtype="torch.float32")
    (saved : Backing layout initial.native c.logits 4) : Backing layout b.computed.native b.work 4 := by
  have first : Backing layout b.allocated.native b.first 4 := by
    have call := b.firstCall
    by_cases inplace : c.options.inplace=Bool.true
    · simp only [if_pos inplace,Step.ok.injEq] at call
      have prepared := saved.retain (snapshot_lifetime layout rt kept life _ _ _ _ b.snapshotCall) heldLogits
      simpa only [← call.1,← call.2] using prepared
    · exact physical.emptyLike _ _ _ _ _ dtype (by simpa only [if_neg inplace] using call)
  have work : Backing layout b.prepared.native b.work 4 := by
    have call := b.workCall
    by_cases packed : lastStride b.first≠1
    · exact physical.emptyLike _ _ _ _ _ dtype (by simpa only [if_pos packed] using call)
    · simp only [if_neg packed,Step.ok.injEq] at call
      simpa only [← call.1,← call.2] using first
  exact work.retain (launch_lifetime layout rt kept life _ (fun _ _ call => .setupBackward call) _ _ _ _ k.launch) heldWork

theorem backward_return_backing {α σ : Type} [Inhabited α] (layout : Layout σ) (rt : Runtime α σ)
    (kept : RegionName → Prop) (life : LifetimeLibrary layout rt kept) (physical : BackingLibrary layout rt)
    {obs : Observations α σ} {objects : ObjectRegistry rt} {M : FP32 α} {c : Context}
    {callerUpstream output : Tensor} {initial final : World α σ}
    {b : BackwardData M rt c callerUpstream output initial final} (k : BackwardKernelData obs objects b)
    (heldLogits : kept c.logits.region) (heldWork : kept b.work.region)
    (dtype : c.logits.dtype="torch.float32") (outputDtype : output.dtype="torch.float32")
    (saved : Backing layout initial.native c.logits 4) : Backing layout final.native output 4 := by
  rcases b.finish with ⟨same,state⟩|⟨different,result,call⟩
  · have backing := backward_work_backing layout rt kept life physical k heldLogits heldWork dtype saved
    have identity := objects.identity b.work output k.work.buffer.registered k.outputObject same
    simpa only [state,identity] using backing
  · exact physical.copy _ _ _ _ _ outputDtype call

#axiomsClean forward_return_backing
#axiomsClean forward_saved_backing
#axiomsClean backward_work_backing
#axiomsClean backward_return_backing
#specNonCircular BackingLibrary avoiding [Source.forward,Source.backward,Grid.forward,Grid.backward,Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Physical
