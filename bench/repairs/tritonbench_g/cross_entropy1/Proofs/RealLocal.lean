import bench.repairs.tritonbench_g.cross_entropy1.Proofs.RealHost

namespace VeriTile.Bench.Repairs.CrossEntropy1.RealHost
open VeriTile.Triton Source Host
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 6000000

/-- Values in the actual rank-local reduction outputs, before collectives.
The source entry is tied to the caller's shard, including target ownership. -/
theorem local_pair_values {σ : Type} (obs : Observations (WithBot ℝ) σ) (rt : Runtime (WithBot ℝ) σ)
    (lib : Library obs rt) (ops : TensorLibrary obs arithmetic rt)
    {o : Options} {logits labels : Tensor} {initial final : World (WithBot ℝ) σ} {returned : ForwardReturn}
    {e : ForwardData RealSource.algebra rt o logits labels initial returned final} (k : ForwardKernelData obs e)
    (trace : AggregateData rt o e.world e.rows e.count e.labels e.losses e.lse e.computed e.pair e.aggregated)
    (dtype : logits.dtype="torch.float32") (labelDtype : labels.dtype="torch.int64")
    (split : e.world>1 ∨ e.cols>65536) (smooth : 0≤o.smoothing)
    (row : Nat) (hr : row<e.rows) (x : Nat → ℝ)
    (caller : ∀ col, col<e.cols → value obs initial logits (row*e.cols+col)=↑(x (e.rank*e.cols+col)))
    (valid : obs.integer initial labels row=o.ignored ∨
      (0≤obs.integer initial labels row ∧ obs.integer initial labels row<(e.world*e.cols : Nat))) :
    value obs trace.localState trace.localPair.1 row=
      ↑(if obs.integer initial labels row=o.ignored then 0 else
        Aggregation.rankContribution e.world e.cols (obs.integer initial labels row).toNat o.smoothing x e.rank) ∧
    value obs trace.localState trace.localPair.2 row=↑(Math.lse e.cols (fun col => x (e.rank*e.cols+col))) := by
  have arrays := forward_arrays obs rt lib k
  have locals := local_reduction_values obs arithmetic rt ops e.count e.rows e.losses e.lse e.computed trace.localPair trace.localState
    k.lossValues k.lseValues arrays.1 arrays.2 k.separation.outputDistinct.symm trace.localCalls
  have lossBound : row<trace.localPair.1.numel := by simpa [Tensor.numel,locals.losses.shapeEq] using hr
  have lseBound : row<trace.localPair.2.numel := by simpa [Tensor.numel,locals.lse.shapeEq] using hr
  have lossReduction := local_sum e.count e.rows row (forward_positive e).2 k.lossValues
    (fun block => RealSource.partialLoss e.parameters (fun col => x (e.rank*e.cols+col)) (obs.integer initial labels row) block)
    (fun block _ => source_loss_slot obs k dtype labelDtype row block hr (fun col => x (e.rank*e.cols+col)) caller split smooth)
  have lseReduction := local_lse e.count e.rows row (forward_positive e).2 k.lseValues
    (fun block => Math.blockLSE e.cols (fwdWidth e.cols) block (fun col => x (e.rank*e.cols+col)))
    (fun block hb => source_lse_slot obs k dtype row block hr hb (fun col => x (e.rank*e.cols+col)) caller)
  constructor
  · rw [locals.losses.values row lossBound,lossReduction]
    apply congrArg (fun r : ℝ => (↑r : WithBot ℝ))
    by_cases ignored : obs.integer initial labels row=o.ignored
    · simp only [RealSource.partialLoss,ForwardData.parameters,forwardParameters,ignored,↓reduceIte,Finset.sum_const_zero]
    · rw [if_neg ignored]
      have math := Aggregation.actual_rank_contribution e.parameters e.world e.rank x (obs.integer initial labels row)
        (valid.resolve_left ignored).1 ignored (forward_width e.cols (forward_positive e).1).1
      have sameParameters : Aggregation.rankParameters e.parameters e.world e.rank=e.parameters := rfl
      rw [sameParameters] at math
      exact math
  · rw [locals.lse.values row lseBound,lseReduction]
    exact congrArg (fun r : ℝ => (↑r : WithBot ℝ))
      (Math.local_lse_composition e.cols (fwdWidth e.cols) (forward_width e.cols (forward_positive e).1).1
        (fun col => x (e.rank*e.cols+col)))

#axiomsClean local_pair_values
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.RealHost
