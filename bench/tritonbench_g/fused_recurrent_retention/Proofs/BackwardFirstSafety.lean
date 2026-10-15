import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardBounds

/-! Safety follows every reached first-loop iteration from the initial
snapshot invariant, with allocation obligations only for active lanes. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardSafety
open VeriTile.Triton Primitives BackwardSource BackwardFirstState SafetyAccess
open ForwardState (keyPointers valuePointers)
noncomputable section
set_option maxHeartbeats 1200000
set_option linter.unusedSimpArgs false

theorem first_iteration_safe (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (hasSeed : Bool) (bounds : RegionBounds) (hb : Bounds p l r hasSeed bounds)
    (t : Nat) (ht : t < p.T) (u : BlockState) (hu : Ready p l initial r hasSeed t u) :
    Stmt.TraceSafeList bounds (firstBody p) u := by
  unfold firstBody
  apply TraceProofs.assign_cons
  · apply masked_load bounds "p_k" "mask_bk" u _ _ hu.k hu.maskK
    rintro ⟨k,⟨⟩⟩ hi
    exact hb.key t k ht (by simpa only [maskTile,Tile.vec,decide_eq_true_eq] using hi)
  intro k
  apply TraceProofs.assign_cons
  · apply masked_load bounds "p_v" "mask_bv" _ (valuePointers p l r.v t)
      (maskTile p.BV (fun j => activeValue p l j.val)) (by simp [hu.v]) (by simp [hu.maskV])
    rintro ⟨j,⟨⟩⟩ hi
    exact hb.value t j ht (by simpa only [maskTile,Tile.vec,decide_eq_true_eq] using hi)
  intro v
  apply TraceProofs.assign_cons
  · apply masked_load bounds "p_do" "mask_bv" _ (valuePointers p l r.do_ t)
      (maskTile p.BV (fun j => activeValue p l j.val)) (by simp [hu.dout]) (by simp [hu.maskV])
    rintro ⟨j,⟨⟩⟩ hi
    exact hb.dout t j ht (by simpa only [maskTile,Tile.vec,decide_eq_true_eq] using hi)
  intro dout
  apply TraceProofs.assign_cons
  · simp [stateUpdate,Op.SafeAt,Op.SafeAt.eq_def]
  intro h
  apply TraceProofs.assign_cons
  · simp [dqProduct,Op.SafeAt,Op.SafeAt.eq_def]
  intro product
  apply TraceProofs.assign_cons
  · simp [Op.SafeAt,Op.SafeAt.eq_def]
  intro output
  apply Stmt.TraceSafeList.cons_intro
  · apply masked_store bounds "p_dq" "d_q" "mask_bk" _ (keyGradientPointers p l r.dq t)
      (maskTile p.BK (fun k => activeKey p l k.val)) (by simp [hu.dq]) (by simp [hu.maskK])
    rintro ⟨k,⟨⟩⟩ hi
    exact hb.dq t k ht (by simpa only [maskTile,Tile.vec,decide_eq_true_eq] using hi)
  intro next _
  apply TraceProofs.all_safe
  simp [ForwardSource.advance,Stmt.TraceSafe,Op.SafeAt,Op.SafeAt.eq_def]

theorem first_loop_safe (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (hasSeed : Bool) (bounds : RegionBounds) (hb : Bounds p l r hasSeed bounds)
    (hd : BackwardReal.Disjoint r) (u : BlockState)
    (hu : BackwardFirstLoop.Holds p l initial r hasSeed 0 u) :
    Stmt.TraceSafe bounds (.forRange "i" 0 p.T 1 (firstBody p)) u := by
  rw [Stmt.TraceSafe]
  apply TraceProofs.range_safe (P := BackwardFirstLoop.Holds p l initial r hasSeed)
  · intro t v ht hv
    exact first_iteration_safe p l initial r hasSeed bounds hb t ht _
      (ready_set_loop p l initial r hasSeed t t v hv.ready)
  · intro t v w ht hv he
    obtain ⟨w',he',hw⟩ := BackwardFirstLoop.iteration p l initial r hasSeed t v
      hd.q_dq hd.k_dq hd.v_dq hd.do_dq hv
    rw [he] at he'
    cases he'
    exact hw
  · exact hu

#axiomsClean first_iteration_safe
#axiomsClean first_loop_safe
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardSafety
