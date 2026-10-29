import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.Calls
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
noncomputable section

/-- Independent dense whole-task mathematics, including O, final state,
dQ/dK/dV, return metadata, saved context, detach and caller frames. A backward
call is supplied after forward and may have any admitted nonzero F. -/
specification correct (p : PlanBinding s) (call : ForwardCall ℝ s caller) :
    ForwardRealPost call (call.sourceRun realArithmetic p) ∧
    ∀ (backCaller : Set Nat)
      (back : BackwardCall ℝ s (call.sourceRun realArithmetic p).context call.before call.currentDevice backCaller),
      BackwardRealPost back (back.sourceRun realArithmetic p) := by
  constructor
  · rw [call.sourceRun_eq]
    exact forward_real p call
  · intro backCaller back
    rw [back.sourceRun_eq]
    exact backward_real call.scope.shape p back

#axiomsClean correct
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention
