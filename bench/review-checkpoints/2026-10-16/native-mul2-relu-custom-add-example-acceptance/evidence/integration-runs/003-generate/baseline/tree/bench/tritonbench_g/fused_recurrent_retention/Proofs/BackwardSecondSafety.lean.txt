import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardBounds

/-! Safety of every reached reverse-loop load and both stores. The second
store follows the actual first store, so its pointer and mask are preserved
by that execution rather than assumed in an unrelated successor state. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardSafety
open VeriTile.Triton Primitives BackwardSource BackwardSecondState SafetyAccess
open ForwardState (keyPointers valuePointers outputPointers)
open BackwardFirstState (keyGradientPointers)
noncomputable section
set_option maxHeartbeats 1500000
set_option linter.unusedSimpArgs false

theorem second_prefix_safe (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (hasSeed : Bool) (bounds : RegionBounds) (hb : Bounds p l r hasSeed bounds)
    (t : Nat) (ht : t < p.T) (u : BlockState) (hu : Ready p l initial r (t+1) u) :
    Stmt.TraceSafeList bounds ((secondBody p).take 8) u := by
  dsimp only [secondBody,List.take]
  apply TraceProofs.assign_cons
  · apply masked_load bounds "p_do" "mask_bv" u (valuePointers p l r.do_ t) _
      (by simpa using hu.dout (by omega)) hu.maskV
    rintro ⟨j,⟨⟩⟩ hi
    exact hb.dout t j ht (by simpa only [maskTile,Tile.vec,decide_eq_true_eq] using hi)
  intro dout
  apply TraceProofs.assign_cons
  · rw [Op.SafeAt.eq_def]
    refine ⟨?_,by simp [Op.SafeAt]⟩
    apply masked_load bounds "p_q" "mask_bk" _ (keyPointers p l r.q t)
      (maskTile p.BK (fun k => activeKey p l k.val))
      (by simpa using hu.q (by omega)) (by simp [hu.maskK])
    rintro ⟨k,⟨⟩⟩ hi
    exact hb.query t k ht (by simpa only [maskTile,Tile.vec,decide_eq_true_eq] using hi)
  intro q
  apply TraceProofs.assign_cons
  · apply masked_load bounds "p_k" "mask_bk" _ (keyPointers p l r.k t)
      (maskTile p.BK (fun k => activeKey p l k.val))
      (by simpa using hu.k (by omega)) (by simp [hu.maskK])
    rintro ⟨k,⟨⟩⟩ hi
    exact hb.key t k ht (by simpa only [maskTile,Tile.vec,decide_eq_true_eq] using hi)
  intro k
  apply TraceProofs.assign_cons
  · apply masked_load bounds "p_v" "mask_bv" _ (valuePointers p l r.v t)
      (maskTile p.BV (fun j => activeValue p l j.val))
      (by simpa using hu.v (by omega)) (by simp [hu.maskV])
    rintro ⟨j,⟨⟩⟩ hi
    exact hb.value t j ht (by simpa only [maskTile,Tile.vec,decide_eq_true_eq] using hi)
  intro v
  apply TraceProofs.all_safe
  simp [dStateUpdate,dkOp,dvOp,Stmt.TraceSafe,Op.SafeAt,Op.SafeAt.eq_def]

theorem second_iteration_safe (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (hasSeed : Bool) (bounds : RegionBounds) (hb : Bounds p l r hasSeed bounds)
    (t : Nat) (ht : t < p.T) (u : BlockState)
    (h : BackwardSecondLoop.Holds p l initial r hasSeed (t+1) u) :
    Stmt.TraceSafeList bounds (secondBody p) u := by
  rw [←List.take_append_drop 8 (secondBody p)]
  apply TraceProofs.append_safe
  · exact second_prefix_safe p l initial r hasSeed bounds hb t ht u h.ready
  · intro mid he
    rw [BackwardSecondValues.before_stores p l initial r t ht u h.ready
      h.qmem h.kmem h.vmem h.domem] at he
    cases he
    dsimp only [secondBody,List.drop]
    rw [TraceProofs.cons_of_run_iff (BackwardSecondStep.key_store p l initial r t u h.ready)]
    refine ⟨?_,?_⟩
    · apply masked_store bounds "p_dk" "d_k" "mask_bk" _ (keyGradientPointers p l r.dk t)
        (maskTile p.BK (fun k => activeKey p l k.val))
        (by simpa [beforeStores] using h.ready.dk (by omega)) (by simp [beforeStores,h.ready.maskK])
      rintro ⟨k,⟨⟩⟩ hi
      exact hb.dk t k ht (by simpa only [maskTile,Tile.vec,decide_eq_true_eq] using hi)
    · apply Stmt.TraceSafeList.cons_intro
      · apply masked_store bounds "p_dv" "d_v" "mask_bv" _ (outputPointers p l r.dv t)
          (maskTile p.BV (fun j => activeValue p l j.val))
          (by simpa [scatterK,beforeStores] using h.ready.dv (by omega))
          (by simp [scatterK,beforeStores,h.ready.maskV])
        rintro ⟨j,⟨⟩⟩ hi
        exact hb.dv t j ht (by simpa only [maskTile,Tile.vec,decide_eq_true_eq] using hi)
      · intro next _
        apply TraceProofs.all_safe
        simp [retreat,Stmt.TraceSafe,Op.SafeAt,Op.SafeAt.eq_def]

theorem holds_set_loop (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (hasSeed : Bool) (remaining i : Nat) (u : BlockState)
    (h : BackwardSecondLoop.Holds p l initial r hasSeed remaining u) :
    BackwardSecondLoop.Holds p l initial r hasSeed remaining (u.setReg "_t" .nat [] (Tile.scalar i)) :=
  ⟨ready_set_loop p l initial r remaining i u h.ready,h.qmem,h.kmem,h.vmem,h.domem,
    h.dq,h.dk,h.dv,h.frame⟩

theorem second_loop_safe (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (hasSeed : Bool) (bounds : RegionBounds) (hb : Bounds p l r hasSeed bounds)
    (hd : BackwardReal.Disjoint r) (u : BlockState)
    (hu : BackwardSecondLoop.Holds p l initial r hasSeed p.T u) :
    Stmt.TraceSafe bounds (.forRange "_t" 0 p.T 1 (secondBody p)) u := by
  rw [Stmt.TraceSafe]
  apply TraceProofs.range_safe (P := fun cur => BackwardSecondLoop.Holds p l initial r hasSeed (p.T-cur))
  · intro cur v hcur hv
    have heq : p.T-cur = (p.T-(cur+1))+1 := by omega
    rw [heq] at hv
    exact second_iteration_safe p l initial r hasSeed bounds hb (p.T-(cur+1)) (by omega) _
      (holds_set_loop p l initial r hasSeed _ cur v hv)
  · intro cur v w hcur hv he
    have heq : p.T-cur = (p.T-(cur+1))+1 := by omega
    rw [heq] at hv
    obtain ⟨w',he',hw⟩ := BackwardSecondLoop.iteration p l initial r hasSeed
      (p.T-(cur+1)) cur (by omega) v hd.toDisjoint hv
    rw [he] at he'
    cases he'
    exact hw
  · simpa using hu

#axiomsClean second_prefix_safe
#axiomsClean second_iteration_safe
#axiomsClean holds_set_loop
#axiomsClean second_loop_safe
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardSafety
