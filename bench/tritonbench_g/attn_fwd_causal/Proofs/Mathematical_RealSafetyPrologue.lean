import bench.tritonbench_g.attn_fwd_causal.Proofs.Mathematical_RealSafetyParts

/-! The source prologue derives its pointers from program IDs and strides.
Only the masked query load and the unmasked query-scale load access memory.
Their bounds are ordinary input-window inequalities. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealSafetyPrologue
open VeriTile.Triton RealLoads RealAccessSafety
open AttentionFwdTriton3.RealAccess AttentionFwdTriton3.RealRegisterSafety
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

theorem init_safe (bounds : RegionBounds) (s : BlockState) (c : Config) (Out : RegionName) (BM BN BD : Nat)
    (hp : RealPointers.Registers c Out BM BN BD s)
    (hbQ : ∀ (r : Fin BM) (d : Fin BD), c.startM*BM+r.val < c.context → d.val < c.activeHead →
      c.base+(c.startM*BM+r.val)*c.strideM+d.val*c.strideD < bounds c.Q)
    (hbS : c.queryScaleBase+c.startM < bounds c.QS) :
    Stmt.TraceSafeList bounds (RealInit.code BM BD c.context c.activeHead) s := by
  unfold RealInit.code
  apply TraceProofs.assign_cons (by simp [Op.SafeAt.eq_def])
  intro m
  apply TraceProofs.assign_cons (by simp [Op.SafeAt.eq_def])
  intro l
  apply TraceProofs.assign_cons (by simp [Op.SafeAt.eq_def])
  intro acc
  apply TraceProofs.assign_cons
  · apply query_safe
    · simpa using hp.query
    · simpa using hp.rows
    · exact hbQ
  · intro q
    apply Stmt.TraceSafeList.cons_intro
    · rw [Stmt.TraceSafe]
      apply query_scale_safe
      · simpa using hp.queryScale
      · exact hbS
    · intro _ _; exact Stmt.TraceSafeList.nil_intro

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealSafetyPrologue
