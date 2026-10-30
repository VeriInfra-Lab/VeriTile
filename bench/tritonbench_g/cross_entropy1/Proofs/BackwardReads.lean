import bench.tritonbench_g.cross_entropy1.Proofs.BackwardKernel
import bench.tritonbench_g.cross_entropy1.Proofs.Mask

namespace VeriTile.Bench.Repairs.CrossEntropy1.Host
open VeriTile.Triton FP.Structural Source
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 4000000

theorem backward_logit_read {α σ : Type} [Inhabited α] (obs : Observations α σ) {rt : Runtime α σ}
    {objects : ObjectRegistry rt} {M : FP32 α} {c : Context} {callerUpstream output : Tensor}
    {initial final : World α σ} {e : BackwardData M rt c callerUpstream output initial final}
    (k : BackwardKernelData obs objects e) (ctx : BackwardContext rt objects c initial) (layout : RowLayout c.logits)
    (row col : Nat) (hr : row<e.rows) (hc : col<e.cols) :
    (k.launch.ready.memory e.parameters.logits (logitAddress e.parameters row col)).read .real=
      value obs initial c.logits (row*e.cols+col) := by
  have coordinate := layout e.rows e.cols e.shape row col hr hc
  have memory := k.initialFrame.memory c.logits.region (c.logits.storageOffset+row*rowStride c.logits+col) ctx.logitsLive
  simp only [value,ctx.logitsNotNeg,Bool.false_eq_true,↓reduceIte,coordinate,
    BackwardData.parameters,backwardParameters,forwardParameters,logitAddress,memory]

theorem backward_upstream_read {α σ : Type} [Inhabited α] (obs : Observations α σ) {rt : Runtime α σ}
    {objects : ObjectRegistry rt} {M : FP32 α} {c : Context} {callerUpstream output : Tensor}
    {initial final : World α σ} {e : BackwardData M rt c callerUpstream output initial final}
    (k : BackwardKernelData obs objects e) (shape : callerUpstream.shape=[e.rows])
    (dtype : callerUpstream.dtype="torch.float32") (row : Nat) (hr : row<e.rows) :
    (k.launch.ready.memory e.parameters.upstream (e.parameters.upstreamBase+row*e.parameters.upstreamStride)).read .real=
      value obs initial callerUpstream row := by
  have outputShape := k.snapshot.materialized.metadata.shape.trans shape
  have coordinate := vector_layout e.upstream e.rows row outputShape k.snapshot.materialized.rank hr
  have inputBound : row<callerUpstream.numel := by simpa [Tensor.numel,shape] using hr
  have logical := k.snapshot.materialized.floats dtype row inputBound
  rw [← k.snapshotFrame.floatValue e.upstream k.snapshot.materialized.live row] at logical
  simpa only [value,k.snapshot.dense.1,Bool.false_eq_true,↓reduceIte,coordinate,
    BackwardData.parameters,backwardParameters] using logical

theorem backward_lse_read {α σ : Type} [Inhabited α] (obs : Observations α σ) {rt : Runtime α σ}
    {objects : ObjectRegistry rt} {M : FP32 α} {c : Context} {callerUpstream output : Tensor}
    {initial final : World α σ} {e : BackwardData M rt c callerUpstream output initial final}
    (k : BackwardKernelData obs objects e) (saved : FloatArray obs initial c.lse [e.rows] fun row => value obs initial c.lse row)
    (dense : c.lse.Dense) (row : Nat) (hr : row<e.rows) :
    (k.launch.ready.memory e.parameters.lse (e.parameters.lseBase+row)).read .real=value obs initial c.lse row := by
  have bound : row<c.lse.numel := by simpa [Tensor.numel,saved.shapeEq] using hr
  have base := dense_base c.lse dense (by omega)
  have frame := k.initialFrame.memory c.lse.region row saved.live
  simp only [BackwardData.parameters,backwardParameters,forwardParameters,base,Nat.zero_add,frame,
    value,dense.1,Bool.false_eq_true,↓reduceIte,dense.2 row bound]

theorem backward_label_read {α σ : Type} [Inhabited α] (obs : Observations α σ) (ints : IntegerViews obs)
    {rt : Runtime α σ} {objects : ObjectRegistry rt} {M : FP32 α} {c : Context} {callerUpstream output : Tensor}
    {initial final : World α σ} {e : BackwardData M rt c callerUpstream output initial final}
    (k : BackwardKernelData obs objects e) (targets : Nat → Int)
    (saved : IntArray obs initial c.labels [e.rows] targets) (dense : c.labels.Dense) (row : Nat) (hr : row<e.rows) :
    Source.label e.parameters k.launch.ready.memory row=targets row := by
  have bound : row<c.labels.numel := by simpa [Tensor.numel,saved.shapeEq] using hr
  have base := dense_base c.labels dense (by omega)
  have frame := k.initialFrame.memory c.labels.region row saved.live
  have label := (ints.dense initial c.labels row saved.dtype dense bound).symm.trans (saved.values row bound)
  simpa only [Source.label,BackwardData.parameters,backwardParameters,forwardParameters,
    Region.cast_cast,Region.cast_id,base,Nat.zero_add,frame] using label

#axiomsClean backward_logit_read
#axiomsClean backward_upstream_read
#axiomsClean backward_lse_read
#axiomsClean backward_label_read
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Host
