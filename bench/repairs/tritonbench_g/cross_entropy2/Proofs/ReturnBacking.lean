import bench.repairs.tritonbench_g.cross_entropy2.Proofs.KernelFrames
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.ReturnBacking

namespace VeriTile.Bench.Repairs.CrossEntropy2.Physical
open VeriTile.Triton Source Host Frames
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Step Observations)
open VeriTile.Bench.Repairs.CrossEntropy1.Physical (Layout Retains Backing)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 6000000

/-- Ordinary backing guarantees for actual single library calls. No result
value, source address, reduction tree, or wrapper execution is assumed. -/
structure BackingLibrary {α σ : Type} (layout : Layout σ) (rt : Runtime α σ) : Prop where
  empty : ∀ shape device before output after,
    rt.empty shape "torch.float32" device before=.ok output after → Backing layout after.native output 4
  emptyLike : ∀ input packed before output after, input.dtype="torch.float32" →
    rt.emptyLike input packed before=.ok output after → Backing layout after.native output 4
  logsumexp : ∀ input before output after, output.dtype="torch.float32" →
    rt.logsumexp0 input before=.ok output after → Backing layout after.native output 4
  mask : ∀ input flags before output after, output.dtype="torch.float32" →
    rt.maskedFillZero input flags before=.ok output after → Backing layout after.native output 4

variable {α σ : Type} {obs : Observations α σ} {layout : Layout σ} {kept : RegionName → Prop}
  {rt : Runtime α σ}

theorem fresh_retains (lib : LocalLibrary obs layout kept rt) {before after : World α σ}
    {output : Tensor} {result : Step (World α σ) Tensor}
    (kind : FreshCall rt before result) (call : result=.ok output after) :
    Retains layout kept before.native after.native := by
  have facts := lib.fresh before result kind
  rw [call] at facts
  exact facts.1.lifetime

theorem write_retains (lib : LocalLibrary obs layout kept rt) {before after : World α σ}
    {dst output : Tensor} {result : Step (World α σ) Tensor}
    (kind : WriteCall rt dst before result) (call : result=.ok output after) :
    Retains layout kept before.native after.native := by
  have facts := lib.write dst before result kind
  rw [call] at facts
  exact facts.2.lifetime

theorem forward_launch_retains [Inhabited α] (lib : LocalLibrary obs layout kept rt)
    (M : FP32 α) (p : Parameters) (count : Nat) (before after : World α σ)
    (call : forwardLaunch M rt p count before=.ok () after) :
    Retains layout kept before.native after.native := by
  unfold forwardLaunch at call
  split at call
  · cases call; exact Retains.refl _ _ _
  · obtain ⟨u,ready,setup,done⟩ := (bind_success _ _ _ _).mp call
    have facts := lib.setup before _ (.forward p count)
    rw [setup] at facts
    cases done
    exact facts.lifetime

theorem backward_launch_retains [Inhabited α] (lib : LocalLibrary obs layout kept rt)
    (M : FP32 α) (p : Parameters) (count : Nat) (before after : World α σ)
    (call : backwardLaunch M rt p count before=.ok () after) :
    Retains layout kept before.native after.native := by
  unfold backwardLaunch at call
  split at call
  · cases call; exact Retains.refl _ _ _
  · obtain ⟨u,ready,setup,done⟩ := (bind_success _ _ _ _).mp call
    have facts := lib.setup before _ (.backward p count)
    rw [setup] at facts
    cases done
    exact facts.lifetime

theorem local_lse_backing (lib : LocalLibrary obs layout kept rt)
    (physical : BackingLibrary layout rt) (count : Nat) (losses lse : Tensor)
    (initial reduced : World α σ) (pair : Tensor × Tensor)
    (trace : VeriTile.Bench.Repairs.CrossEntropy1.Host.LocalTrace rt.toRuntime count losses lse initial pair reduced)
    (held : kept pair.2.region) (dtype : pair.2.dtype="torch.float32")
    (original : kept lse.region → Backing layout initial.native lse 4) :
    Backing layout reduced.native pair.2 4 := by
  cases trace with
  | direct one => exact original held
  | reduced many localLSE localLoss logged summed lseCall sumCall =>
    exact (physical.logsumexp _ _ _ _ dtype lseCall).retain
      (fresh_retains lib (.sum losses) sumCall) held

theorem aggregate_return_backing (lib : LocalLibrary obs layout kept rt)
    (physical : BackingLibrary layout rt) {o : Options} {count : Nat}
    {labels losses lse : Tensor} {initial final : World α σ} {returned : Return}
    (e : AggregateData rt o count labels losses lse initial returned final)
    (labelsDtype : labels.dtype="torch.int32")
    (lossDtype : returned.1.dtype="torch.float32") (lseDtype : returned.2.1.dtype="torch.float32")
    (zDtype : returned.2.2.dtype="torch.float32")
    (heldLoss : kept returned.1.region) (heldLSE : kept returned.2.1.region)
    (originalLSE : kept lse.region → Backing layout initial.native lse 4) :
    Backing layout final.native returned.1 4 ∧ Backing layout final.native returned.2.1 4 ∧
    Backing layout final.native returned.2.2 4 := by
  have pairHeld : kept e.pair.2.region := by simpa only [e.lseIdentity] using heldLSE
  have pairDtype : e.pair.2.dtype="torch.float32" := by simpa only [e.lseIdentity] using lseDtype
  have reduced := local_lse_backing lib physical count losses lse initial e.reduced e.pair
    e.localCalls pairHeld pairDtype originalLSE
  have afterReduction := (write_retains lib (.add e.pair.2) e.addCall).trans
    ((fresh_retains lib (.scale o.squareScale e.pair.2) e.scaleCall).trans
    ((fresh_retains lib (.multiply e.scaled e.pair.2) e.multiplyCall).trans
    ((write_retains lib (.add e.zLoss) e.correctionCall).trans
    ((fresh_retains lib (.widen labels labelsDtype) e.widenCall).trans
    ((fresh_retains lib (.compare e.wideLabels o.ignored) e.compareCall).trans
    ((write_retains lib (.mask e.mask) e.maskLossCall).trans
      (write_retains lib (.mask e.mask) e.maskZCall)))))))
  refine ⟨(physical.mask _ _ _ _ _ lossDtype e.maskLossCall).retain
    (write_retains lib (.mask e.mask) e.maskZCall) heldLoss,?_,
    physical.mask _ _ _ _ _ zDtype e.maskZCall⟩
  simpa only [e.lseIdentity] using reduced.retain afterReduction pairHeld

/-- All three actual forward returns have complete four-byte coordinate
backing, derived from allocations, reductions and final mask operations. -/
theorem forward_return_backing [Inhabited α] (lib : LocalLibrary obs layout kept rt)
    (physical : BackingLibrary layout rt)
    {M : FP32 α} {o : Options} {x y : Tensor} {rows cols : Nat}
    {initial final : World α σ} {returned : Return}
    {e : ForwardData M rt o x y rows cols initial returned final} (k : ForwardKernelData obs e)
    (lossDtype : returned.1.dtype="torch.float32") (lseDtype : returned.2.1.dtype="torch.float32")
    (zDtype : returned.2.2.dtype="torch.float32")
    (heldLoss : kept returned.1.region) (heldLSE : kept returned.2.1.region) (heldZ : kept returned.2.2.region) :
    Backing layout final.native returned.1 4 ∧ Backing layout final.native returned.2.1 4 ∧
    Backing layout final.native returned.2.2 4 := by
  have launch := forward_launch_retains lib M _ _ _ _ e.launchCall
  have lseLife := (fresh_retains lib (.empty _ _ _) e.zCall).trans launch
  have lseBacking : kept e.lse.region → Backing layout e.computed.native e.lse 4 :=
    fun held => (physical.empty _ _ _ _ _ e.lseCall).retain lseLife held
  have finish := e.finishCall
  cases split : e.split with
  | «false» =>
    simp only [split,Bool.false_eq_true,↓reduceIte,Step.ok.injEq] at finish
    have returnedEq : returned=(e.loss,e.lse,e.zLoss) := finish.1.symm
    have lossBacking := (physical.empty _ _ _ _ _ e.lossCall).retain
      ((fresh_retains lib (.empty _ _ _) e.lseCall).trans lseLife)
      (show kept e.loss.region by simpa only [returnedEq] using heldLoss)
    have saved := lseBacking (by simpa only [returnedEq] using heldLSE)
    have zBacking := (physical.empty _ _ _ _ _ e.zCall).retain launch
      (show kept e.zLoss.region by simpa only [returnedEq] using heldZ)
    simpa only [returnedEq,← finish.2] using And.intro lossBacking (And.intro saved zBacking)
  | «true» =>
    simp only [split,↓reduceIte] at finish
    obtain ⟨trace⟩ := aggregate_success rt o e.count e.labels e.loss e.lse e.computed returned final finish
    exact aggregate_return_backing lib physical trace
      (k.labels.materialized.metadata.dtype.trans e.admitted.2.2.1)
      lossDtype lseDtype zDtype heldLoss heldLSE lseBacking

/-- The standalone backward return is the actual first fresh allocation,
whose backing is retained through the successful setup and grid. -/
theorem backward_return_backing [Inhabited α] (lib : LocalLibrary obs layout kept rt)
    (physical : BackingLibrary layout rt)
    {M : FP32 α} {o : Options} {g x l y : Tensor} {rows cols : Nat}
    {initial final : World α σ} {returned : Tensor}
    {e : BackwardData M rt o g x l y rows cols initial returned final} (k : BackwardKernelData obs e)
    (held : kept returned.region) : Backing layout final.native returned 4 :=
  (physical.emptyLike _ _ _ _ _ (k.logits.materialized.metadata.dtype.trans e.admitted.2.1) e.allocationCall).retain
    (backward_launch_retains lib M _ _ _ _ e.launchCall) held

#axiomsClean aggregate_return_backing
#axiomsClean forward_return_backing
#axiomsClean backward_return_backing
#specNonCircular BackingLibrary avoiding [Host.forward,Host.backward,Host.aggregate,Source.forward,Source.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Physical
