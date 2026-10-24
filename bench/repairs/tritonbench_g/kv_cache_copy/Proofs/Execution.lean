import bench.repairs.tritonbench_g.kv_cache_copy.Proofs.Memory

namespace VeriTile.Bench.Repairs.KvCacheCopy
noncomputable section
open Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 100000

theorem scatter_frame {ι α : Type} (events : List ι) (key : ι → Nat) (value : ι → α) (before : Memory α)
    (x : Nat) (miss : ∀ e∈events, x≠key e) : scatter events key value before x=before x := by
  induction events generalizing before with
  | nil => rfl
  | cons e tail ih =>
    change scatter tail key value (Function.update before (key e) (value e)) x=before x
    rw [ih _ (fun f hf => miss f (by simp [hf])),Function.update_of_ne (miss e (by simp))]

theorem scatter_stable {ι α : Type} (events : List ι) (key : ι → Nat) (value : ι → α) (before : Memory α)
    (x : Nat) (v : α) (initial : before x=v) (agree : ∀ e∈events, key e=x → value e=v) :
    scatter events key value before x=v := by
  induction events generalizing before with
  | nil => exact initial
  | cons e tail ih =>
    apply ih
    · change Function.update before (key e) (value e) x=v
      by_cases eq : key e=x
      · rw [eq,Function.update_self]; exact agree e (by simp) eq
      · rw [Function.update_of_ne (Ne.symm eq)]; exact initial
    · intro f hf; exact agree f (by simp [hf])

theorem scatter_read {ι α : Type} (events : List ι) (key : ι → Nat) (value : ι → α) (before : Memory α)
    (e : ι) (member : e∈events) (agree : ∀ f∈events, key f=key e → value f=value e) :
    scatter events key value before (key e)=value e := by
  obtain ⟨a,b,he⟩ := List.append_of_mem member
  rw [he]
  unfold scatter
  rw [List.foldl_append,List.foldl_cons]
  apply scatter_stable
  · simp
  · intro f hf eq
    exact agree f (by rw [he]; simp [hf]) eq

theorem scatter_append {ι α : Type} (a b : List ι) (key : ι → Nat) (value : ι → α) (heap : Memory α) :
    scatter (a++b) key value heap=scatter b key value (scatter a key value heap) := by
  exact List.foldl_append

theorem scatter_values {ι α : Type} (events : List ι) (key : ι → Nat) (a b : ι → α)
    (agree : ∀ e∈events,a e=b e) (heap : Memory α) : scatter events key a heap=scatter events key b heap := by
  induction events generalizing heap with
  | nil => rfl
  | cons e rest ih =>
    unfold scatter
    simp only [List.foldl_cons,agree e (by simp)]
    exact ih (fun x hx => agree x (by simp [hx])) _

theorem launch_frozen_aux (m : Inputs) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m b l t)
    (work : List Tile) (valid : ∀ e∈work,ValidTile (config m b) e) (initial heap : Memory α)
    (same : ∀ e,ValidLane (config m b) e → heap (sourceAddress m (config m b) e)=initial (sourceAddress m (config m b) e)) :
    work.foldl (tile m (config m b) l t) heap=
      scatter (work.flatMap (tileLanes (config m b))) (targetAddress m (config m b) l t)
        (fun e => initial (sourceAddress m (config m b) e)) heap := by
  induction work generalizing heap with
  | nil => rfl
  | cons x rest ih =>
    have hx := valid x (by simp)
    have laneValid : ∀ e∈tileLanes (config m b) x,ValidLane (config m b) e := by
      intro e he
      obtain ⟨eq,hc⟩ := (tileLanes_member (config m b) x e).mp he
      exact ⟨eq ▸ hx,hc⟩
    have keep : ∀ e,ValidLane (config m b) e →
        tile m (config m b) l t heap x (sourceAddress m (config m b) e)=initial (sourceAddress m (config m b) e) := by
      intro e he
      apply Eq.trans (scatter_frame _ _ _ _ _ ?_) (same e he)
      intro f hf
      exact source_target_separate m b l t h e f he (laneValid f hf)
    have frozen : tile m (config m b) l t heap x=
        scatter (tileLanes (config m b) x) (targetAddress m (config m b) l t)
          (fun e => initial (sourceAddress m (config m b) e)) heap :=
      scatter_values _ _ _ _ (fun e he => same e (laneValid e he)) heap
    simp only [List.foldl_cons,List.flatMap_cons]
    rw [ih (fun e he => valid e (by simp [he])) _ keep,scatter_append,frozen]

theorem launch_frozen (m : Inputs) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m b l t)
    (before : Memory α) :
    launch m (config m b) l t before=
      scatter (lanes (config m b)) (targetAddress m (config m b) l t)
        (fun e => before (sourceAddress m (config m b) e)) before :=
  launch_frozen_aux m b l t h _ (fun e he => (tiles_member _ _).mp he) before before (by intros;rfl)

theorem launch_correct (m : Inputs) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m b l t)
    (before : Memory α) : CacheCopy m (config m b) l t before (launch m (config m b) l t before) := by
  rw [launch_frozen m b l t h before]
  constructor
  · intro key s head d hs hh hd
    have geometry := split_geometry m b l t h
    let e : Lane := ⟨⟨s,head,d/(config m b).width⟩,key,d%(config m b).width⟩
    have valid : ValidLane (config m b) e := by
      refine ⟨⟨hs,hh,?_⟩,Nat.mod_lt _ geometry.1⟩
      exact (Nat.div_lt_iff_lt_mul geometry.1).mpr (by simpa [geometry.2] using hd)
    have dim : component (config m b) e=d := by
      simpa only [component,e,Nat.mul_comm] using Nat.div_add_mod d (config m b).width
    have read := scatter_read (lanes (config m b)) (targetAddress m (config m b) l t)
      (fun f => before (sourceAddress m (config m b) f)) before e ((lanes_member _ _).mpr valid) (by
        intro f hf same
        rw [target_injective m b l t h f e ((lanes_member _ _).mp hf) valid same])
    simpa only [targetAddress,sourceAddress,dim,e] using read
  · intro a outside
    apply scatter_frame
    intro e he
    have valid := (lanes_member (config m b) e).mp he
    exact outside e.key e.tile.sequence e.tile.head (component (config m b) e)
      valid.1.1 valid.1.2.1 (component_bound _ (split_geometry m b l t h).2 e valid)

#axiomsClean launch_frozen
#axiomsClean launch_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KvCacheCopy
