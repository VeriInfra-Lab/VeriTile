import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.ReturnedSplit
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.RealLocal

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Host
open CrossEntropy1.Host (World Observations TensorLibrary value fwdWidth forward_width local_reduction_values)
open CrossEntropy1.RealHost (arithmetic block_coordinates local_sum local_lse)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

theorem source_lse_slot {σ : Type} (obs : Observations (WithBot ℝ) σ) {rt : Runtime (WithBot ℝ) σ}
    {o : Options} {logits labels : Tensor} {initial final : World (WithBot ℝ) σ} {returned : ForwardReturn}
    {e : ForwardData RealSource.algebra rt o logits labels initial returned final} (k : ForwardKernelData obs e)
    (dtype : logits.dtype="torch.float32") (row block : Nat) (hr : row<e.rows) (hb : block<e.count) (x : Nat → ℝ)
    (caller : ∀ col, col<e.cols → value obs initial logits (row*e.cols+col)=↑(x col)) :
    k.lseValues (block*e.rows+row)=↑(CrossEntropy1.Math.blockLSE e.cols (fwdWidth e.cols) block (Spec.scaled o.logitScale x)) := by
  have coordinates := block_coordinates e.rows row block hr
  simp only [ForwardKernelData.lseValues,coordinates.1,coordinates.2]
  exact RealSource.forward_lse e.parameters k.launch.ready.memory row block x (caller_input obs k dtype row hr x caller)
    (forward_width e.cols (forward_positive e).1).1
    (CrossEntropy1.Math.block_starts e.cols (fwdWidth e.cols) block (forward_width e.cols (forward_positive e).1).1 hb)

theorem source_loss_slot {σ : Type} (obs : Observations (WithBot ℝ) σ) {rt : Runtime (WithBot ℝ) σ}
    {o : Options} {logits labels : Tensor} {initial final : World (WithBot ℝ) σ} {returned : ForwardReturn}
    {e : ForwardData RealSource.algebra rt o logits labels initial returned final} (k : ForwardKernelData obs e)
    (dtype : logits.dtype="torch.float32") (labelDtype : labels.dtype="torch.int64")
    (row block : Nat) (hr : row<e.rows) (x : Nat → ℝ)
    (caller : ∀ col, col<e.cols → value obs initial logits (row*e.cols+col)=↑(x col))
    (split : e.world>1 ∨ e.cols>65536) (smooth : 0≤o.smoothing) :
    k.lossValues (block*e.rows+row)=
      ↑(CrossEntropy1.RealSource.partialLoss e.parameters.toParameters (Spec.scaled o.logitScale x) (obs.integer initial labels row) block) := by
  have coordinates := block_coordinates e.rows row block hr
  simp only [ForwardKernelData.lossValues,coordinates.1,coordinates.2,forward_kernel_labels obs k labelDtype row hr]
  apply RealSource.forward_split_loss e.parameters k.launch.ready.memory row block x (caller_input obs k dtype row hr x caller)
  · simp [ForwardData.parameters,forwardParameters,CrossEntropy1.Host.forwardParameters,split]
  · intro noSmooth
    have h : ¬0<o.smoothing := by
      simpa only [ForwardData.parameters,forwardParameters,CrossEntropy1.Host.forwardParameters,decide_eq_false_iff_not] using noSmooth
    change o.smoothing=0
    linarith

/-- Values in the actual rank-local reduction outputs, before collectives.
The source entry is tied to the caller's shard, including target ownership. -/
theorem local_pair_values {σ : Type} (obs : Observations (WithBot ℝ) σ) (rt : Runtime (WithBot ℝ) σ)
    (lib : LocalAPI obs rt) (ops : TensorLibrary obs arithmetic rt.toRuntime)
    {o : Options} {logits labels : Tensor} {initial final : World (WithBot ℝ) σ} {returned : ForwardReturn}
    {e : ForwardData RealSource.algebra rt o logits labels initial returned final} (k : ForwardKernelData obs e)
    (trace : AggregateData rt o e.world e.rows e.count e.labels e.losses e.lse e.computed e.result e.aggregated)
    (dtype : logits.dtype="torch.float32") (labelDtype : labels.dtype="torch.int64")
    (split : e.world>1 ∨ e.cols>65536) (smooth : 0≤o.smoothing)
    (row : Nat) (hr : row<e.rows) (x : Nat → ℝ)
    (caller : ∀ col, col<e.cols → value obs initial logits (row*e.cols+col)=↑(x (e.rank*e.cols+col)))
    (valid : obs.integer initial labels row=o.ignored ∨
      (0≤obs.integer initial labels row ∧ obs.integer initial labels row<(e.world*e.cols : Nat))) :
    value obs trace.localState trace.localPair.1 row=
      ↑(if obs.integer initial labels row=o.ignored then 0 else
        CrossEntropy1.Aggregation.rankContribution e.world e.cols (obs.integer initial labels row).toNat o.smoothing (Spec.scaled o.logitScale x) e.rank) ∧
    value obs trace.localState trace.localPair.2 row=↑(Spec.lse e.cols o.logitScale (fun col => x (e.rank*e.cols+col))) := by
  have arrays := forward_arrays obs rt lib k
  have locals := local_reduction_values obs arithmetic rt.toRuntime ops e.count e.rows e.losses e.lse e.computed trace.localPair trace.localState
    k.lossValues k.lseValues arrays.1 arrays.2 k.separation.outputDistinct.symm trace.localCalls
  have lossBound : row<trace.localPair.1.numel := by simpa [Tensor.numel,locals.losses.shapeEq] using hr
  have lseBound : row<trace.localPair.2.numel := by simpa [Tensor.numel,locals.lse.shapeEq] using hr
  have lossReduction := local_sum e.count e.rows row (forward_positive e).2 k.lossValues
    (fun block => CrossEntropy1.RealSource.partialLoss e.parameters.toParameters (Spec.scaled o.logitScale (fun col => x (e.rank*e.cols+col))) (obs.integer initial labels row) block)
    (fun block _ => source_loss_slot obs k dtype labelDtype row block hr (fun col => x (e.rank*e.cols+col)) caller split smooth)
  have lseReduction := local_lse e.count e.rows row (forward_positive e).2 k.lseValues
    (fun block => CrossEntropy1.Math.blockLSE e.cols (fwdWidth e.cols) block (Spec.scaled o.logitScale (fun col => x (e.rank*e.cols+col))))
    (fun block hb => source_lse_slot obs k dtype row block hr hb (fun col => x (e.rank*e.cols+col)) caller)
  constructor
  · rw [locals.losses.values row lossBound,lossReduction]
    apply congrArg (fun r : ℝ => (↑r : WithBot ℝ))
    by_cases ignored : obs.integer initial labels row=o.ignored
    · simp only [CrossEntropy1.RealSource.partialLoss,ForwardData.parameters,forwardParameters,CrossEntropy1.Host.forwardParameters,ignored,↓reduceIte,Finset.sum_const_zero]
    · rw [if_neg ignored]
      have math := CrossEntropy1.Aggregation.actual_rank_contribution e.parameters.toParameters e.world e.rank (Spec.scaled o.logitScale x) (obs.integer initial labels row)
        (valid.resolve_left ignored).1 ignored (forward_width e.cols (forward_positive e).1).1
      have sameParameters : CrossEntropy1.Aggregation.rankParameters e.parameters.toParameters e.world e.rank=e.parameters.toParameters := rfl
      rw [sameParameters] at math
      exact math
  · rw [locals.lse.values row lseBound,lseReduction]
    exact congrArg (fun r : ℝ => (↑r : WithBot ℝ))
      (CrossEntropy1.Math.local_lse_composition e.cols (fwdWidth e.cols) (forward_width e.cols (forward_positive e).1).1
        (Spec.scaled o.logitScale (fun col => x (e.rank*e.cols+col))))

#axiomsClean source_lse_slot
#axiomsClean source_loss_slot
#axiomsClean local_pair_values
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Host
