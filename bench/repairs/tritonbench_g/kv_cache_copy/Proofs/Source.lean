import bench.repairs.tritonbench_g.kv_cache_copy.Proofs.Native

namespace VeriTile.Bench.Repairs.KvCacheCopy
noncomputable section
open Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 100000

theorem scatter_congr {ι α : Type} (events : List ι) (k j : ι → Nat) (a b : ι → α)
    (keys : ∀ e∈events,k e=j e) (values : ∀ e∈events,a e=b e) (heap : Memory α) :
    scatter events k a heap=scatter events j b heap := by
  induction events generalizing heap with
  | nil => rfl
  | cons e rest ih =>
    unfold scatter
    simp only [List.foldl_cons,keys e (by simp),values e (by simp)]
    exact ih (fun e he => keys e (by simp [he])) (fun e he => values e (by simp [he])) _

theorem tile_readonly (m : Inputs) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m b l t)
    (heap : Memory α) (x : Tile) (hx : ValidTile (config m b) x)
    (r : Role) (readonly : r≠.keyCache ∧ r≠.valueCache) (a : Nat) (inside : Inside m r a) :
    tile m (config m b) l t heap x a=heap a := by
  apply scatter_frame
  intro e he same
  obtain ⟨eq,hc⟩ := (tileLanes_member (config m b) x e).mp he
  have target := target_bound m b l t h e ⟨eq ▸ hx,hc⟩
  rw [←same] at target
  exact separate_inside m h.separate r _ (by cases e.key <;> simp [outputRole,readonly]) a inside target

theorem tile_binding (m : Inputs) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m b l t)
    (decode : α → Word) (heap : Memory α) (binding : TensorBinding m b l t decode heap)
    (x : Tile) (hx : ValidTile (config m b) x) :
    TensorBinding m b l t decode (tile m (config m b) l t heap x) := by
  have sizes := numel m b l t h
  refine ⟨binding.capacity,?_,?_⟩
  · intro s hs
    rw [tile_readonly m b l t h heap x hx .lengths (by decide) _
      (in_address m .lengths s (by rw [sizes.2.2.2.2.1];exact hs))]
    exact binding.lengths s hs
  · intro s c hs hc
    rw [tile_readonly m b l t h heap x hx .table (by decide) _
      (in_address m .table _ (by rw [sizes.2.2.2.2.2];exact Linear.linear_bound _ _ _ _ hs hc))]
    exact binding.table s c hs hc

/-- Generic mutable fold transport; the invariant is ordinary Tensor readback. -/
theorem foldl_equiv {α ι : Type} (f g : Memory α → ι → Memory α) (P : Memory α → Prop) (Q : ι → Prop)
    (step : ∀ heap,P heap → ∀ x,Q x → f heap x=g heap x)
    (keep : ∀ heap,P heap → ∀ x,Q x → P (g heap x))
    (work : List ι) (valid : ∀ x∈work,Q x) (heap : Memory α) (hp : P heap) :
    work.foldl f heap=work.foldl g heap := by
  induction work generalizing heap with
  | nil => rfl
  | cons x rest ih =>
    have hx := valid x (by simp)
    simp only [List.foldl_cons]
    rw [step heap hp x hx]
    exact ih (fun y hy => valid y (by simp [hy])) _ (keep heap hp x hx)

theorem foldl_binding (m : Inputs) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m b l t)
    (decode : α → Word) (work : List Tile) (valid : ∀ x∈work,ValidTile (config m b) x)
    (heap : Memory α) (binding : TensorBinding m b l t decode heap) :
    TensorBinding m b l t decode (work.foldl (tile m (config m b) l t) heap) := by
  induction work generalizing heap with
  | nil => exact binding
  | cons x rest ih =>
    exact ih (fun y hy => valid y (by simp [hy])) _
      (tile_binding m b l t h decode heap binding x (valid x (by simp)))

namespace Native

theorem tile_exact (m : Inputs) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m b l t)
    (decode : α → Word) (heap : Memory α) (binding : TensorBinding m b l t decode heap)
    (x : Tile) (hx : ValidTile (config m b) x) :
    tile m (config m b) (i (blockId (config m b) l t x.sequence),i (past l x.sequence)) heap x=
      KvCacheCopy.tile m (config m b) l t heap x := by
  apply scatter_congr
  · intro e he
    obtain ⟨eq,hc⟩ := (tileLanes_member (config m b) x e).mp he
    have pointer := target_pointer m b l t h decode heap binding e ⟨eq ▸ hx,hc⟩
    simpa only [eq] using pointer
  · intro e he
    obtain ⟨eq,hc⟩ := (tileLanes_member (config m b) x e).mp he
    rw [source_pointer m b l t h decode heap binding e ⟨eq ▸ hx,hc⟩]

def programTiles (p : Config) (id : Nat × Nat) : List Tile :=
  (List.range p.splits).map fun q => ⟨id.1,id.2,q⟩

theorem programTiles_valid (p : Config) (id : Nat × Nat) (ids : id.1<p.count ∧ id.2<p.heads)
    (x : Tile) (member : x∈programTiles p id) : ValidTile p x ∧ x.sequence=id.1 := by
  obtain ⟨q,hq,rfl⟩ := List.mem_map.mp member
  exact ⟨⟨ids.1,ids.2,List.mem_range.mp hq⟩,rfl⟩

theorem program_exact (m : Inputs) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m b l t)
    (decode : α → Word) (heap : Memory α) (binding : TensorBinding m b l t decode heap)
    (id : Nat × Nat) (ids : id.1<(config m b).count ∧ id.2<(config m b).heads) :
    program m (config m b) decode heap id=
      (programTiles (config m b) id).foldl (KvCacheCopy.tile m (config m b) l t) heap := by
  unfold program
  rw [resolve_exact m b l t h decode heap binding id.1 ids.1]
  exact foldl_equiv _ _ (TensorBinding m b l t decode)
    (fun x => ValidTile (config m b) x ∧ x.sequence=id.1)
    (fun state hb x hx => by simpa only [hx.2] using tile_exact m b l t h decode state hb x hx.1)
    (fun state hb x hx => tile_binding m b l t h decode state hb x hx.1)
    _ (programTiles_valid (config m b) id ids) heap binding

theorem programs_member (p : Config) (id : Nat × Nat) : id∈programs p ↔ id.1<p.count ∧ id.2<p.heads := by
  cases id
  simp [programs]

theorem flatten_programs (p : Config) : (programs p).flatMap (programTiles p)=tiles p := by
  simp [programs,programTiles,tiles,List.flatMap_assoc,List.flatMap_map]

theorem launch_exact (m : Inputs) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m b l t)
    (decode : α → Word) (heap : Memory α) (binding : TensorBinding m b l t decode heap) :
    launch m (config m b) decode heap=KvCacheCopy.launch m (config m b) l t heap := by
  have transport := foldl_equiv (program m (config m b) decode)
    (fun heap id => (programTiles (config m b) id).foldl (KvCacheCopy.tile m (config m b) l t) heap)
    (TensorBinding m b l t decode) (fun id => id.1<(config m b).count ∧ id.2<(config m b).heads)
    (fun state hb id hi => program_exact m b l t h decode state hb id hi)
    (fun state hb id hi => foldl_binding m b l t h decode _
      (fun x hx => (programTiles_valid (config m b) id hi x hx).1) state hb)
    (programs (config m b)) (fun id hi => (programs_member _ _).mp hi) heap binding
  rw [←List.foldl_flatMap,flatten_programs] at transport
  exact transport

#axiomsClean program_exact
#axiomsClean launch_exact
end Native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KvCacheCopy
