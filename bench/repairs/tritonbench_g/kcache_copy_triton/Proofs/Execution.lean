import bench.repairs.tritonbench_g.kcache_copy_triton.Proofs.Memory
import bench.repairs.tritonbench_g.kv_cache_copy.Proofs.Execution
namespace VeriTile.Bench.Repairs.KcacheCopyTriton
noncomputable section
open Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 100000
export KvCacheCopy (scatter_frame scatter_stable scatter_read scatter_append scatter_values)

theorem launch_frozen_aux (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m n b l t)
    (work : List Tile) (valid : ∀ e∈work,ValidTile (config m n b) e) (initial heap : Memory α)
    (same : ∀ e,ValidLane (config m n b) e → heap (sourceAddress m (config m n b) e)=initial (sourceAddress m (config m n b) e)) :
    work.foldl (tile m (config m n b) l t) heap=
      scatter (work.flatMap (tileLanes (config m n b))) (targetAddress m (config m n b) l t)
        (fun e => initial (sourceAddress m (config m n b) e)) heap := by
  induction work generalizing heap with
  | nil => rfl
  | cons x rest ih =>
    have hx := valid x (by simp)
    have laneValid : ∀ e∈tileLanes (config m n b) x,ValidLane (config m n b) e := by
      intro e he
      obtain ⟨eq,hc⟩ := (tileLanes_member (config m n b) x e).mp he
      exact ⟨eq ▸ hx,hc⟩
    have keep : ∀ e,ValidLane (config m n b) e →
        tile m (config m n b) l t heap x (sourceAddress m (config m n b) e)=initial (sourceAddress m (config m n b) e) := by
      intro e he
      apply Eq.trans (scatter_frame _ _ _ _ _ ?_) (same e he)
      intro f hf
      exact source_target_separate m n b l t h e f he (laneValid f hf)
    have frozen : tile m (config m n b) l t heap x=
        scatter (tileLanes (config m n b) x) (targetAddress m (config m n b) l t)
          (fun e => initial (sourceAddress m (config m n b) e)) heap :=
      scatter_values _ _ _ _ (fun e he => same e (laneValid e he)) heap
    simp only [List.foldl_cons,List.flatMap_cons]
    rw [ih (fun e he => valid e (by simp [he])) _ keep]
    simp only [scatter] at frozen ⊢
    rw [scatter_append,frozen]

theorem launch_frozen (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m n b l t)
    (before : Memory α) :
    launch m (config m n b) l t before=
      scatter (lanes (config m n b)) (targetAddress m (config m n b) l t)
        (fun e => before (sourceAddress m (config m n b) e)) before :=
  launch_frozen_aux m n b l t h _ (fun e he => (tiles_member _ _).mp he) before before (by intros;rfl)

theorem launch_correct (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m n b l t)
    (before : Memory α) : CacheCopy m (config m n b) l t before (launch m (config m n b) l t before) := by
  rw [launch_frozen m n b l t h before]
  constructor
  · intro s head d hs hh hd
    have geometry := split_geometry m n b l t h
    let e : Lane := ⟨⟨s,head,d/(config m n b).width⟩,d%(config m n b).width⟩
    have valid : ValidLane (config m n b) e := by
      refine ⟨⟨hs,hh,?_⟩,Nat.mod_lt _ geometry.1⟩
      exact (Nat.div_lt_iff_lt_mul geometry.1).mpr (by simpa [geometry.2] using hd)
    have dim : component (config m n b) e=d := by
      simpa only [component,e,Nat.mul_comm] using Nat.div_add_mod d (config m n b).width
    have read := scatter_read (lanes (config m n b)) (targetAddress m (config m n b) l t)
      (fun f => before (sourceAddress m (config m n b) f)) before e ((lanes_member _ _).mpr valid) (by
        intro f hf same
        rw [target_injective m n b l t h f e ((lanes_member _ _).mp hf) valid same])
    simpa only [targetAddress,sourceAddress,dim,e] using read
  · intro a outside
    apply scatter_frame
    intro e he
    have valid := (lanes_member (config m n b) e).mp he
    exact outside e.tile.sequence e.tile.head (component (config m n b) e)
      valid.1.1 valid.1.2.1 (component_bound _ (split_geometry m n b l t h).2 e valid)

#axiomsClean launch_frozen
#axiomsClean launch_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KcacheCopyTriton
