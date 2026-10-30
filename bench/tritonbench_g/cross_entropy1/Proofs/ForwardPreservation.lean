import bench.tritonbench_g.cross_entropy1.Proofs.ForwardArrays
import bench.tritonbench_g.cross_entropy1.Proofs.Mask

namespace VeriTile.Bench.Repairs.CrossEntropy1.Host
open VeriTile.Triton FP.Structural Source
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 4000000

theorem forward_grid_region {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters)
    (entry : Memory α) (count : Nat) (separation : Grid.ForwardSeparation p)
    (r : RegionName) (lossSeparate : r≠p.loss) (lseSeparate : r≠p.lse) (i : Nat) :
    Grid.forward M p entry count r i=entry r i := by
  apply Grid.forward_complete_frame M p entry count separation
  intro q hq written
  rcases written with h|h
  · exact lossSeparate h.1
  · exact lseSeparate h.1

theorem forward_launch_preserves {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (lib : Library obs rt) (ints : IntegerViews obs) {M : FP32 α} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs e)
    (before : World α σ) (frame : Preserves obs before k.launch.ready)
    (lossFresh : ¬before.live e.losses.region) (lseFresh : ¬before.live e.lse.region) :
    Preserves obs before e.computed := by
  let computed := {k.launch.ready with memory:=Grid.forward M e.parameters k.launch.ready.memory e.count}
  have unchanged : ∀ r, before.live r → ∀ i, computed.memory r i=k.launch.ready.memory r i := by
    intro r live i
    apply forward_grid_region M e.parameters k.launch.ready.memory e.count k.separation
    · intro same
      apply lossFresh
      simpa only [ForwardData.parameters,forwardParameters] using (same ▸ live : before.live e.parameters.loss)
    · intro same
      apply lseFresh
      simpa only [ForwardData.parameters,forwardParameters] using (same ▸ live : before.live e.parameters.lse)
  have core : Preserves obs before computed := by
    refine ⟨frame.live,?_,?_⟩
    · intro r i live
      exact (unchanged r live i).trans (frame.memory r i live)
    · intro t i live
      have kernelRead := ints.localRead computed k.launch.ready t i rfl (unchanged t.region live)
      exact kernelRead.trans (frame.integers t i live)
  exact core.trans (lib.exit _ _ _ k.launch.exitCall)

theorem forward_resolved_frame {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (lib : Library obs rt) (ints : IntegerViews obs) {M : FP32 α} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs e) :
    Preserves obs e.resolved e.computed := by
  apply forward_launch_preserves obs rt lib ints k e.resolved k.workingFrame k.lossAllocation.fresh
  intro live
  exact k.lseAllocation.fresh (k.lossAllocation.frame.live _ live)

theorem forward_caller_frame {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (lib : Library obs rt) (ints : IntegerViews obs) {M : FP32 α} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs e) :
    Preserves obs initial e.computed := k.working.frame.trans (forward_resolved_frame obs rt lib ints k)

theorem forward_label_array {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (lib : Library obs rt) (ints : IntegerViews obs) {M : FP32 α} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs e)
    (dtype : labels.dtype="torch.int64") :
    IntArray obs e.computed e.labels [e.rows] (obs.integer initial labels) := by
  have exit := lib.exit _ _ _ k.launch.exitCall
  have outputDtype := k.labelsSnapshot.materialized.metadata.dtype.trans dtype
  have outputShape := k.labelsSnapshot.materialized.metadata.shape.trans e.labelsShape
  refine ⟨outputShape,outputDtype,exit.live _ k.labelsLive,?_⟩
  intro row hr
  have inputBound : row<labels.numel := by simpa only [k.labelsSnapshot.materialized.metadata.numel] using hr
  rw [ints.dense e.computed e.labels row outputDtype k.labelsSnapshot.dense hr]
  have frame := forward_grid_region M e.parameters k.launch.ready.memory e.count k.separation e.labels.region
    (by simpa only [ForwardData.parameters,forwardParameters,Region.cast_cast,Region.cast_id] using k.separation.lossLabels.symm)
    (by simpa only [ForwardData.parameters,forwardParameters,Region.cast_cast,Region.cast_id] using k.separation.lseLabels.symm) row
  have memory := (exit.memory e.labels.region row k.labelsLive).trans
    (frame.trans (k.labelsFrame.memory _ _ k.labelsSnapshot.materialized.live))
  have cell : (e.computed.memory e.labels.region row).read .int=obs.integer initial labels row :=
    (congrArg (fun c : Source.Cell α => c.read .int) memory).trans (k.labelsSnapshot.labels dtype row inputBound)
  exact cell

#axiomsClean forward_launch_preserves
#axiomsClean forward_caller_frame
#axiomsClean forward_label_array
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Host
