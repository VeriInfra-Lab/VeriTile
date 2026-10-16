import bench.tritonbench_g.fused_recurrent_retention.Proofs.SafetyAccess
import bench.tritonbench_g.fused_recurrent_retention.Proofs.ForwardLoop

/-! Safety along every reached source loop iteration, using the actual source
pointer advances and independent initial-snapshot invariant. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.ForwardLoopSafety
open VeriTile.Triton ForwardState ForwardSource Primitives SafetyAccess
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

theorem iteration_safe (p : Layout) (l : Launch) (initial : BlockState) (r : ForwardRegions)
    (hasSeed storeFinal : Bool) (bounds : RegionBounds) (hb : Bounds p l r hasSeed storeFinal bounds)
    (t : Nat) (ht : t < p.T) (u : BlockState) (hu : Ready p l initial r hasSeed t u) :
    Stmt.TraceSafeList bounds (loopBody p) u := by
  unfold loopBody
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
  · rw [Op.SafeAt.eq_def]
    refine ⟨?_,by simp [Op.SafeAt]⟩
    apply masked_load bounds "p_q" "mask_bk" _ (keyPointers p l r.q t)
      (maskTile p.BK (fun j => activeKey p l j.val)) (by simp [hu.q]) (by simp [hu.maskK])
    rintro ⟨k,⟨⟩⟩ hi
    exact hb.query t k ht (by simpa only [maskTile,Tile.vec,decide_eq_true_eq] using hi)
  intro q
  apply TraceProofs.assign_cons
  · simp [stateUpdate,Op.SafeAt,Op.SafeAt.eq_def]
  intro h
  apply TraceProofs.assign_cons
  · simp [outputProduct,Op.SafeAt,Op.SafeAt.eq_def]
  intro product
  apply TraceProofs.assign_cons
  · simp [Op.SafeAt,Op.SafeAt.eq_def]
  intro output
  apply Stmt.TraceSafeList.cons_intro
  · apply masked_store bounds "p_o" "_o" "mask_bv" _ (outputPointers p l r.o t)
      (maskTile p.BV (fun j => activeValue p l j.val)) (by simp [hu.o]) (by simp [hu.maskV])
    rintro ⟨j,⟨⟩⟩ hi
    exact hb.output t j ht (by simpa only [maskTile,Tile.vec,decide_eq_true_eq] using hi)
  intro next _
  apply TraceProofs.all_safe
  simp [advance,Stmt.TraceSafe,Op.SafeAt,Op.SafeAt.eq_def]

theorem loop_safe (p : Layout) (l : Launch) (initial : BlockState) (r : ForwardRegions)
    (hasSeed storeFinal : Bool) (bounds : RegionBounds) (hb : Bounds p l r hasSeed storeFinal bounds)
    (hqo : r.q ≠ r.o) (hko : r.k ≠ r.o) (hvo : r.v ≠ r.o)
    (u : BlockState) (hu : ForwardLoop.Holds p l initial r hasSeed 0 u) :
    Stmt.TraceSafe bounds (.forRange "_t" 0 p.T 1 (loopBody p)) u := by
  rw [Stmt.TraceSafe]
  apply TraceProofs.range_safe (P := ForwardLoop.Holds p l initial r hasSeed)
  · intro t v ht hv
    exact iteration_safe p l initial r hasSeed storeFinal bounds hb t ht _
      (ready_set_loop p l initial r hasSeed t t v hv.ready)
  · intro t v w ht hv he
    obtain ⟨w',he',hw⟩ := ForwardLoop.iteration p l initial r hasSeed t v hqo hko hvo hv
    rw [he] at he'
    cases he'
    exact hw
  · exact hu

#axiomsClean iteration_safe
#axiomsClean loop_safe
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.ForwardLoopSafety
