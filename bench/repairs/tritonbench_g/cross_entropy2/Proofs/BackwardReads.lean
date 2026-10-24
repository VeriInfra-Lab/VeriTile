import bench.repairs.tritonbench_g.cross_entropy2.Proofs.BackwardKernel

namespace VeriTile.Bench.Repairs.CrossEntropy2.Host
open VeriTile.Triton Source
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Observations Preserves value)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 6000000

theorem Normalized.raw_float {α σ : Type} [Inhabited α] {obs : Observations α σ}
    {input output : Tensor} {before after ready : World α σ}
    (n : Normalized obs input output before after) (frame : Preserves obs after ready)
    (dtype : input.dtype="torch.float32") (i : Nat) (inside : i < input.numel) :
    (ready.memory output.region (output.index i)).read .real=value obs before input i := by
  have read := (frame.floatValue output n.materialized.live i).trans (n.materialized.floats dtype i inside)
  simpa only [value,n.packed.1,Bool.false_eq_true,↓reduceIte] using read

variable {α σ : Type} [Inhabited α] {obs : Observations α σ} {M : FP32 α} {rt : Runtime α σ} {o : Options}
  {g x l y : Tensor} {rows cols : Nat} {initial final : World α σ} {returned : Tensor}
  {e : BackwardData M rt o g x l y rows cols initial returned final}

theorem backward_kernel_logits (k : BackwardKernelData obs e) (shape : x.shape=[rows,cols])
    (row col : Nat) (hr : row<rows) (hc : col<cols) :
    (k.ready.memory e.parameters.logits (BackwardGrid.logitAddress e.parameters row col)).read .real=
      value obs initial x (row*cols+col) := by
  have bound : row*cols+col<x.numel := by
    simp only [Tensor.numel,shape,List.prod_cons,List.prod_nil,Nat.mul_one]
    have h := Nat.mul_le_mul_right cols (Nat.succ_le_of_lt hr)
    simp only [Nat.succ_mul] at h
    omega
  have read := k.logits.raw_float k.logitsFrame e.admitted.2.1 _ bound
  rw [k.logits.rows rows cols (k.logits.materialized.metadata.shape.trans shape) row col hr hc] at read
  exact read

theorem backward_kernel_labels (A : Arithmetic α) (lib : Library obs A rt)
    (k : BackwardKernelData obs e) (yLive : initial.live y.region) (row : Nat) (hr : row<rows) :
    BackwardGrid.label e.parameters k.ready.memory row=obs.integer initial y row := by
  have inputSize : y.numel=rows := by simp [Tensor.numel,e.admitted.2.2.2.1]
  have normalized := k.labels.int32 e.admitted.2.2.1 row (by simpa only [inputSize] using hr)
  have raw := lib.int32Raw e.labels e.normalizedLabels row k.labels.materialized.live
    (k.labels.materialized.metadata.dtype.trans e.admitted.2.2.1) k.labels.packed
    (by simpa only [k.labels.materialized.metadata.numel,inputSize] using hr)
  have frame := k.labelsFrame.memory e.labels.region (e.labels.storageOffset+row) k.labels.materialized.live
  change (k.ready.memory e.labels.region (e.labels.storageOffset+row)).read .int=_
  rw [frame,raw,normalized,k.logits.materialized.frame.integers y row yLive]

theorem backward_kernel_upstream (k : BackwardKernelData obs e) (gLive : initial.live g.region)
    (row : Nat) (hr : row<rows) :
    (k.ready.memory e.parameters.upstream (e.parameters.upstreamBase+row*e.parameters.upstreamStride)).read .real=
      value obs initial g row := by
  have bound : row<g.numel := by simpa [Tensor.numel,e.upstreamShape] using hr
  have read := k.upstream.raw_float k.upstreamFrame e.upstreamDtype row bound
  rw [VeriTile.Bench.Repairs.CrossEntropy1.Host.vector_layout e.upstream rows row
    (k.upstream.materialized.metadata.shape.trans e.upstreamShape) k.upstream.materialized.rank hr] at read
  exact read.trans ((k.logits.materialized.frame.trans k.labels.materialized.frame).floatValue g gLive row)

theorem backward_kernel_lse (k : BackwardKernelData obs e) (lLive : initial.live l.region)
    (row : Nat) (hr : row<rows) :
    (k.ready.memory e.parameters.lse (e.parameters.lseBase+row)).read .real=value obs initial l row := by
  have bound : row<l.numel := by simpa [Tensor.numel,e.lseShape] using hr
  have read := k.lse.raw_float k.lseFrame e.lseDtype row bound
  rw [k.lse.packed.2 row (by simpa only [k.lse.materialized.metadata.numel] using bound)] at read
  exact read.trans (((k.logits.materialized.frame.trans k.labels.materialized.frame).trans k.upstream.materialized.frame).floatValue l lLive row)

#axiomsClean backward_kernel_logits
#axiomsClean backward_kernel_labels
#axiomsClean backward_kernel_upstream
#axiomsClean backward_kernel_lse
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Host
