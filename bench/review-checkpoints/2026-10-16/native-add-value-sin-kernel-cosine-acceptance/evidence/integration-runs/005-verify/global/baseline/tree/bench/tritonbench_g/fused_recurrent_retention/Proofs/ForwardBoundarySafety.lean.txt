import bench.tritonbench_g.fused_recurrent_retention.Proofs.ForwardInitial
import bench.tritonbench_g.fused_recurrent_retention.Proofs.ForwardFinal
import bench.tritonbench_g.fused_recurrent_retention.Proofs.SafetyAccess

/-! Memory safety of the actual seed and final-state branches, with bounds
required only for enabled branches and active key/value lanes. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.ForwardBoundarySafety
open VeriTile.Triton ForwardState StatePrimitives ForwardHeaderSource SafetyAccess
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

theorem header_safe (p : Layout) (r : ForwardRegions) (swapped : Bool)
    (bounds : RegionBounds) (u : BlockState) :
    Stmt.TraceSafeList bounds (header p r swapped) u := by
  apply TraceProofs.all_safe
  cases swapped <;> simp [header,mulRef,decayOp,pointerOp,outputPointerOp,keyMask,
    valueMask,maskKV,Stmt.TraceSafe,Op.SafeAt,Op.SafeAt.eq_def]

theorem seed_safe (p : Layout) (r : ForwardRegions) (hasSeed storeFinal : Bool)
    (initial : BlockState) (bounds : RegionBounds)
    (hb : Bounds p (launch initial) r hasSeed storeFinal bounds) :
    Stmt.TraceSafeList bounds (seedCode p r hasSeed) (ForwardHeader.state p r initial) := by
  apply Stmt.TraceSafeList.cons_intro
  · cases hasSeed
    · simp [Stmt.TraceSafe,Op.SafeAt,evalOp.eq_def,Tile.scalar_data]
    · simp only [Stmt.TraceSafe,Op.SafeAt,evalOp.eq_def,Tile.scalar_data,true_and,↓reduceIte]
      rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
        (HeaderPrimitives.state_pointer p (launch initial) r.initial _
          (by simp [ForwardHeader.state]) (by simp [ForwardHeader.state])
          (by simp [ForwardHeader.state])))]
      refine ⟨by simp [Stmt.TraceSafe,statePointerOp,mulRef,Op.SafeAt,Op.SafeAt.eq_def], ?_⟩
      apply Stmt.TraceSafeList.cons_intro
      · rw [Stmt.TraceSafe,seedAdd,Op.SafeAt.eq_def]
        refine ⟨by simp [Op.SafeAt],?_⟩
        apply masked_load bounds "p_init_s" "mask_kv" _ (statePointers p (launch initial) r.initial)
          (stateMask p (launch initial)) (by simp) (by simp [ForwardHeader.state])
        rintro ⟨j,k,⟨⟩⟩ hi
        have hh : activeKey p (launch initial) k.val ∧ activeValue p (launch initial) j.val := by
          simpa only [stateMask,Tile.mat,decide_eq_true_eq] using hi
        exact hb.seed rfl k j hh.1 hh.2
      · intro _ _; exact Stmt.TraceSafeList.nil_intro
  · intro _ _; exact Stmt.TraceSafeList.nil_intro

theorem initial_safe (p : Layout) (r : ForwardRegions) (hasSeed storeFinal swapped : Bool)
    (initial : BlockState) (bounds : RegionBounds)
    (hb : Bounds p (launch initial) r hasSeed storeFinal bounds) :
    Stmt.TraceSafeList bounds (ForwardSource.initialCode p r hasSeed storeFinal swapped) initial := by
  rw [initial_exact]
  apply TraceProofs.append_safe
  · exact header_safe p r swapped bounds initial
  · intro u hu
    rw [ForwardHeader.executes] at hu
    cases hu
    exact seed_safe p r hasSeed storeFinal initial bounds hb

theorem final_safe (p : Layout) (l : Launch) (initial : BlockState) (r : ForwardRegions)
    (hasSeed storeFinal swapped : Bool) (bounds : RegionBounds)
    (hb : Bounds p l r hasSeed storeFinal bounds) (u : BlockState)
    (hu : Ready p l initial r hasSeed p.T u) :
    Stmt.TraceSafeList bounds (ForwardSource.finalCode p r hasSeed storeFinal swapped) u := by
  rw [final_exact]
  apply Stmt.TraceSafeList.cons_intro
  · cases storeFinal
    · simp [Stmt.TraceSafe,Op.SafeAt,evalOp.eq_def,Tile.scalar_data]
    · simp only [Stmt.TraceSafe,Op.SafeAt,evalOp.eq_def,Tile.scalar_data,true_and,↓reduceIte]
      rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
        (HeaderPrimitives.state_pointer p l r.final u hu.headId hu.keyId hu.valueId))]
      refine ⟨by simp [Stmt.TraceSafe,statePointerOp,mulRef,Op.SafeAt,Op.SafeAt.eq_def], ?_⟩
      apply Stmt.TraceSafeList.cons_intro
      · apply masked_store bounds "p_final_s" "h" "mask_kv" _ (statePointers p l r.final)
          (stateMask p l) (by simp) (by simpa [stateMask] using hu.maskKV)
        rintro ⟨j,k,⟨⟩⟩ hi
        have hh : activeKey p l k.val ∧ activeValue p l j.val := by
          simpa only [stateMask,Tile.mat,decide_eq_true_eq] using hi
        exact hb.finalState rfl k j hh.1 hh.2
      · intro _ _; exact Stmt.TraceSafeList.nil_intro
  · intro _ _; exact Stmt.TraceSafeList.nil_intro

#axiomsClean header_safe
#axiomsClean seed_safe
#axiomsClean initial_safe
#axiomsClean final_safe
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.ForwardBoundarySafety
