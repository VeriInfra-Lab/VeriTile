import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardBounds

/-! Actual header, optional seed and middle-section safety, preserving the
barrier and all pointer rebases while imposing no inactive-lane bounds. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardSafety
open VeriTile.Triton BackwardHeaderSource BackwardHeaderPrimitives SafetyAccess
noncomputable section
set_option maxHeartbeats 1300000
set_option linter.unusedSimpArgs false

theorem header_safe (p : Layout) (r : BackwardRegions) (swapped : Bool)
    (bounds : RegionBounds) (u : BlockState) :
    Stmt.TraceSafeList bounds (header p r swapped) u := by
  apply TraceProofs.all_safe
  cases swapped <;> simp [header,ForwardHeaderSource.mulRef,ForwardHeaderSource.decayOp,
    ForwardHeaderSource.pointerOp,gradientPointerOp,ForwardHeaderSource.keyMask,
    ForwardHeaderSource.valueMask,Stmt.TraceSafe,Op.SafeAt,Op.SafeAt.eq_def]

theorem seed_safe (p : Layout) (r : BackwardRegions) (hasSeed : Bool)
    (initial : BlockState) (bounds : RegionBounds)
    (hb : Bounds p (launch initial) r hasSeed bounds) :
    Stmt.TraceSafeList bounds (seedCode p r hasSeed) (BackwardHeader.state p r initial) := by
  apply Stmt.TraceSafeList.cons_intro
  · cases hasSeed
    · simp [Stmt.TraceSafe,Op.SafeAt,evalOp.eq_def,Tile.scalar_data]
    · simp only [Stmt.TraceSafe,Op.SafeAt,evalOp.eq_def,Tile.scalar_data,true_and,↓reduceIte]
      rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
        (state_mask p (launch initial) _ (by simp [BackwardHeader.state])
          (by simp [BackwardHeader.state])))]
      refine ⟨by simp [Stmt.TraceSafe,maskKV,Op.SafeAt,Op.SafeAt.eq_def],?_⟩
      rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
        (state_pointer p (launch initial) r.initial _ (by simp [BackwardHeader.state])
          (by simp [BackwardHeader.state]) (by simp [BackwardHeader.state])))]
      refine ⟨by simp [Stmt.TraceSafe,statePointerOp,ForwardHeaderSource.mulRef,Op.SafeAt,Op.SafeAt.eq_def],?_⟩
      apply Stmt.TraceSafeList.cons_intro
      · rw [Stmt.TraceSafe,seedAdd,Op.SafeAt.eq_def]
        refine ⟨by simp [Op.SafeAt],?_⟩
        apply masked_load bounds "p_init_s" "mask_kv" _ (statePointers p (launch initial) r.initial)
          (stateMask p (launch initial)) (by simp) (by simp)
        rintro ⟨k,j,⟨⟩⟩ hi
        have hh : activeKey p (launch initial) k.val ∧ activeValue p (launch initial) j.val := by
          simpa only [stateMask,Tile.mat,decide_eq_true_eq] using hi
        exact hb.seed rfl k j hh.1 hh.2
      · intro _ _; exact Stmt.TraceSafeList.nil_intro
  · intro _ _; exact Stmt.TraceSafeList.nil_intro

theorem initial_safe (p : Layout) (r : BackwardRegions) (hasSeed swapped : Bool)
    (initial : BlockState) (bounds : RegionBounds)
    (hb : Bounds p (launch initial) r hasSeed bounds) :
    Stmt.TraceSafeList bounds (BackwardSource.initialCode p r hasSeed swapped) initial := by
  rw [initial_exact]
  apply TraceProofs.append_safe
  · exact header_safe p r swapped bounds initial
  · intro u hu
    rw [BackwardHeader.executes] at hu
    cases hu
    exact seed_safe p r hasSeed initial bounds hb

theorem middle_safe (p : Layout) (r : BackwardRegions) (hasSeed swapped : Bool)
    (bounds : RegionBounds) (u : BlockState) :
    Stmt.TraceSafeList bounds (BackwardSource.middleCode p r hasSeed swapped) u := by
  rw [BackwardMiddleSource.middle_exact]
  apply TraceProofs.all_safe
  simp [BackwardMiddleSource.code,BackwardMiddleSource.pointerOp,
    BackwardMiddleSource.gradientPointerOp,BackwardMiddleSource.outputPointerOp,
    BackwardMiddleSource.lastOffset,ForwardHeaderSource.mulRef,Stmt.TraceSafe,
    Op.SafeAt,Op.SafeAt.eq_def,evalOp.eq_def,Tile.scalar_data,Stmt.TraceSafeList]

#axiomsClean header_safe
#axiomsClean seed_safe
#axiomsClean initial_safe
#axiomsClean middle_safe
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardSafety
