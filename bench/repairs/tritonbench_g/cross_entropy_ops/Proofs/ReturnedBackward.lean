import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.BackwardReads
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.ReturnedUnsplit

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Host
open VeriTile.Triton
open CrossEntropy1.Host (World Observations IntegerViews ObjectRegistry BackwardContext RowLayout FloatArray IntArray value)
open CrossEntropy1.RealHost (block_coordinates)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

/-- Final logical gradients from the saved forward values and caller's actual
upstream snapshot. Public composition supplies the saved LSE via the proved
forward return theorem; it must not assume this value as a fresh oracle. -/
theorem backward_real_values {σ : Type} (obs : Observations (WithBot ℝ) σ) (rt : Runtime (WithBot ℝ) σ)
    (lib : LocalAPI obs rt) (ints : IntegerViews obs) (objects : ObjectRegistry rt.toRuntime)
    {c : Context} {callerUpstream output : Tensor} {initial final : World (WithBot ℝ) σ}
    {e : BackwardData RealSource.algebra rt c callerUpstream output initial final}
    (k : BackwardKernelData obs objects e) (ctx : BackwardContext rt.toRuntime objects c.base initial)
    (layout : RowLayout c.logits) (dtype : c.logits.dtype="torch.float32")
    (lseDense : c.lse.Dense) (labelsDense : c.labels.Dense)
    (start : Nat) (classStart : c.classStart=start) (smooth : 0≤c.options.smoothing)
    (x : Nat → Nat → ℝ) (targets : Nat → Int) (upstream : Nat → ℝ)
    (savedLogits : ∀ row, row<e.rows → ∀ col, col<e.cols →
      value obs initial c.logits (row*e.cols+col)=↑(x row (start+col)))
    (savedLSE : FloatArray obs initial c.lse [e.rows] (fun row => ↑(Spec.lse c.totalClasses c.options.logitScale (x row))))
    (savedLabels : IntArray obs initial c.labels [e.rows] targets)
    (upstreamShape : callerUpstream.shape=[e.rows]) (upstreamDtype : callerUpstream.dtype="torch.float32")
    (caller : ∀ row, row<e.rows → value obs initial callerUpstream row=↑(upstream row)) :
    ∀ row, row<e.rows → ∀ col, col<e.cols →
      value obs final output (row*e.cols+col)=
        ↑(Spec.gradient c.totalClasses c.options.logitScale c.options.smoothing c.options.squareScale
          (upstream row) (x row) (targets row) c.options.ignored (start+col)) := by
  have returned := backward_return_values obs rt lib k dtype
  intro row hr col hc
  have bound : row*e.cols+col<output.numel := by
    simp only [Tensor.numel,k.outputShape,e.shape,List.prod_cons,List.prod_nil,Nat.mul_one]
    have product := Nat.mul_le_mul_right e.cols (Nat.succ_le_of_lt hr)
    simp only [Nat.succ_mul] at product
    omega
  rw [returned.2 _ bound]
  have coordinates := block_coordinates e.cols col row hc
  simp only [BackwardKernelData.gradientValues,coordinates.1,coordinates.2]
  have positive : 0<e.parameters.block := Nat.pos_of_ne_zero e.nonzero
  let block := col/e.parameters.block
  let lane : Fin e.parameters.block := ⟨col%e.parameters.block,Nat.mod_lt col positive⟩
  have columnEq : CrossEntropy1.Source.column e.parameters.toParameters block lane.val=col := by
    simpa only [CrossEntropy1.Source.column,block,lane,Nat.mul_comm e.parameters.block] using Nat.div_add_mod col e.parameters.block
  have input : RealSource.Input e.parameters k.launch.ready.memory row (fun j => x row (start+j)) := by
    intro j hj
    exact (backward_logit_read obs k ctx layout row j hr hj).trans (savedLogits row hr j hj)
  have lseBound : row<c.lse.numel := by simpa [Tensor.numel,savedLSE.shapeEq] using hr
  have lseRead := (backward_lse_read obs k
    ⟨savedLSE.shapeEq,savedLSE.dtype,savedLSE.live,fun _ _ => rfl⟩ lseDense row hr).trans (savedLSE.values row lseBound)
  have upstreamRead := (backward_upstream_read obs k upstreamShape upstreamDtype row hr).trans (caller row hr)
  have lseUpstream : e.parameters.lse≠e.parameters.upstream := by
    change c.lse.region≠e.upstream.region
    intro same
    exact k.snapshot.fresh (same ▸ ctx.lseLive)
  have source := RealSource.backward_lane e.parameters k.launch.ready.memory row block (fun j => x row (start+j)) input
    (targets row) (Spec.lse c.totalClasses c.options.logitScale (x row)) (upstream row)
    (backward_label_read obs ints k targets savedLabels labelsDense row hr) lseRead upstreamRead
    ctx.logitsLSE.symm lseUpstream
    (CrossEntropy1.RealHost.smoothing_zero c.options.toOptions rfl rfl smooth) lane (by simpa only [columnEq] using hc)
  have indicator : (col : Int)=targets row-e.parameters.classStart ↔ ((start+col : Nat) : Int)=targets row := by
    change (col : Int)=targets row-(c.classStart : Int) ↔ ((start+col : Nat) : Int)=targets row
    rw [classStart]
    omega
  simpa only [columnEq,Spec.gradient,indicator] using source

#axiomsClean backward_real_values
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Host
