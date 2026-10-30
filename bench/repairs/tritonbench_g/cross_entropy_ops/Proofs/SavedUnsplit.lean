import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.ForwardViews
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.ForwardPreservation
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.ReturnedUnsplit

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Host
open VeriTile.Triton
open CrossEntropy1.Source (FP32)
open CrossEntropy1.Host (World Observations Preserves Materialized ObjectRegistry BackwardContext
  RowLayout FloatArray IntArray IntegerViews rowStride)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

theorem materialized_preserve {α σ : Type} [Inhabited α] {obs : Observations α σ}
    {input output : Tensor} {before middle after : World α σ}
    (made : Materialized obs input output before middle) (frame : Preserves obs middle after) :
    Materialized obs input output before after := by
  refine ⟨made.metadata,made.frame.trans frame,frame.live _ made.live,?_,?_,made.rank⟩
  · intro dtype i hi
    exact (frame.floatValue output made.live i).trans (made.floats dtype i hi)
  · intro dtype i hi
    exact (frame.integers output i made.live).trans (made.integers dtype i hi)

/-- Metadata and logical contents of the four objects actually saved by
forward. The LSE value is supplied separately by the forward return theorem. -/
structure SavedForwardData {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (objects : ObjectRegistry rt.toRuntime) (logits labels : Tensor) (initial final : World α σ)
    (returned : ForwardReturn) (rows : Nat) : Prop where
  context : BackwardContext rt.toRuntime objects returned.context.base final
  logits : Materialized obs logits returned.context.logits initial final
  layout : RowLayout returned.context.logits
  labels : IntArray obs final returned.context.labels [rows] (obs.integer initial labels)
  labelsDense : returned.context.labels.Dense
  lseShape : returned.context.lse.shape=[rows]
  lseDtype : returned.context.lse.dtype="torch.float32"
  lseDense : returned.context.lse.Dense

theorem unsplit_saved {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (lib : LocalAPI obs rt) (ints : IntegerViews obs) (objects : ObjectRegistry rt.toRuntime)
    {M : FP32 α} {o : Options} {logits labels : Tensor} {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs e)
    (v : ForwardViewData obs objects e) (labelDtype : labels.dtype="torch.int64")
    (unsplit : ¬(e.world>1 ∨ e.cols>65536)) :
    SavedForwardData obs rt objects logits labels initial final returned e.rows := by
  have ids := unsplit_identity e unsplit
  have post : Preserves obs e.computed final := by
    rw [← ids.2.1]
    exact (lib.tensor.save _ _ _ _ _ _ e.saveCall).trans (lib.mark _ _ _ e.markCall)
  have resolved := (forward_resolved_frame obs rt lib ints k).trans post
  have working := materialized_preserve k.working resolved
  have labelsFinal := (forward_label_array obs rt lib ints k labelDtype).preserve post
  have exit := lib.tensor.exit _ _ _ k.launch.exitCall
  have lseLive := post.live _ (exit.live _ k.lseLive)
  have contextEq : returned.context=⟨e.workingLogits,e.lse,e.labels,e.inplaceLogits,o,e.world*e.cols,e.rank*e.cols⟩ := by
    simpa only [ids.1] using congrArg ForwardReturn.context e.returnIdentity
  have ctx : BackwardContext rt.toRuntime objects returned.context.base final := by
    simp only [contextEq,Context.base]
    refine ⟨working.live,labelsFinal.live,lseLive,resolved.live _ (v.afterOriginal.live _ v.inplace.live),
      k.working.rank,k.notNeg,v.workingObject,v.inplaceObject,
      v.inplace.metadata.shape.trans k.working.metadata.shape.symm,v.workingLabels,k.separation.lseLogits.symm,?_⟩
    intro inplace rows cols shape
    have same : [rows,cols]=[e.rows,e.cols] := shape.symm.trans (k.working.metadata.shape.trans e.logitsShape)
    have eqs : rows=e.rows ∧ cols=e.cols := by simpa only [List.cons.injEq,and_true] using same
    simpa only [eqs.1,eqs.2] using v.workingRows inplace
  refine ⟨ctx,?_,?_,?_,?_,?_,?_,?_⟩
  · simpa only [contextEq] using working
  · simpa only [contextEq] using k.layout
  · simpa only [contextEq] using labelsFinal
  · simpa only [contextEq] using k.labelsSnapshot.dense
  · simpa only [contextEq,ids.2.2,gt_iff_lt,Nat.lt_irrefl,↓reduceIte] using k.lseAllocation.shapeEq
  · simpa only [contextEq] using k.lseAllocation.dtypeEq
  · simpa only [contextEq] using k.lseDense

#axiomsClean materialized_preserve
#axiomsClean unsplit_saved
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Host
