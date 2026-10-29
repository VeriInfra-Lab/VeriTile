import bench.repairs.tritonbench_g.kcache_copy_triton.Proofs.Contract
import bench.repairs.tritonbench_g.kv_cache_copy.Proofs.Memory
namespace VeriTile.Bench.Repairs.KcacheCopyTriton
noncomputable section
set_option maxHeartbeats 4000000
namespace Linear
export KvCacheCopy.Linear (linear_bound linear_injective)
end Linear

def ValidTile (p : Config) (e : Tile) : Prop := KvCacheCopy.ValidTile p.toConfig e
def ValidLane (p : Config) (e : Lane) : Prop := ValidTile p e.tile ∧ e.lane<p.width

theorem tiles_member (p : Config) (e : Tile) : e∈tiles p ↔ ValidTile p e := KvCacheCopy.tiles_member p.toConfig e

theorem tileLanes_member (p : Config) (t : Tile) (e : Lane) : e∈tileLanes p t ↔ e.tile=t ∧ e.lane<p.width := by
  cases e
  simp [tileLanes,eq_comm,and_comm]

theorem lanes_member (p : Config) (e : Lane) : e∈lanes p ↔ ValidLane p e := by
  simp only [lanes,List.mem_flatMap,tiles_member,tileLanes_member,ValidLane]
  constructor
  · rintro ⟨t,ht,he,hc⟩;exact ⟨he ▸ ht,hc⟩
  · rintro ⟨ht,hc⟩;exact ⟨e.tile,ht,rfl,hc⟩

theorem component_bound (p : Config) (dims : p.splits*p.width=p.dims) (e : Lane) (he : ValidLane p e) :
    component p e<p.dims := by
  rw [←dims]
  exact Linear.linear_bound _ _ _ _ he.1.2.2 he.2

def Inside (m : Inputs) (r : Role) (a : Nat) : Prop := base m r≤a ∧ a<base m r+4*(m r).numel

theorem in_address (m : Inputs) (r : Role) (i : Nat) (hi : i<(m r).numel) : Inside m r (address m r i) := by
  unfold Inside address
  omega

theorem source_bound (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m n b l t)
    (e : Lane) (he : ValidLane (config m n b) e) : Inside m .key (sourceAddress m (config m n b) e) := by
  apply in_address
  rw [(numel m n b l t h).1]
  exact Linear.linear_bound _ _ _ _ (Linear.linear_bound _ _ _ _ he.1.1 he.1.2.1)
    (component_bound _ (split_geometry m n b l t h).2 e he)

theorem target_bound (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m n b l t)
    (e : Lane) (he : ValidLane (config m n b) e) : Inside m .cache (targetAddress m (config m n b) l t e) := by
  apply in_address
  have geometry := split_geometry m n b l t h
  have bounds := metadata_bounds m n b l t h e.tile.sequence he.1.1
  rw [(numel m n b l t h).2.1]
  exact KvCacheCopy.cache_bound (config m n b).toConfig geometry.1 geometry.2 _ _ _ _ _
    bounds.2.2.2 he.1.2.1 bounds.2.2.1 (component_bound _ geometry.2 e he)

theorem separate_inside (m : Inputs) (separate : Separate m) (r s : Role) (different : r≠s)
    (a : Nat) (hr : Inside m r a) (hs : Inside m s a) : False := by
  rcases separate r s different with before|after <;> unfold Inside at hr hs <;> omega

theorem source_target_separate (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m n b l t)
    (e f : Lane) (he : ValidLane (config m n b) e) (hf : ValidLane (config m n b) f) :
    sourceAddress m (config m n b) e≠targetAddress m (config m n b) l t f := by
  intro same
  have source := source_bound m n b l t h e he
  rw [same] at source
  exact separate_inside m h.separate _ _ (by decide) _ source (target_bound m n b l t h f hf)

theorem target_injective (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m n b l t)
    (e f : Lane) (he : ValidLane (config m n b) e) (hf : ValidLane (config m n b) f)
    (same : targetAddress m (config m n b) l t e=targetAddress m (config m n b) l t f) : e=f := by
  have addressEq : outputIndex (config m n b) l t e.tile.sequence e.tile.head (component (config m n b) e)=
      outputIndex (config m n b) l t f.tile.sequence f.tile.head (component (config m n b) f) := by
    unfold targetAddress address at same
    omega
  have geometry := split_geometry m n b l t h
  have eb := metadata_bounds m n b l t h e.tile.sequence he.1.1
  have fb := metadata_bounds m n b l t h f.tile.sequence hf.1.1
  obtain ⟨blocks,heads,slots,components⟩ := KvCacheCopy.cache_injective (config m n b).toConfig geometry.1 geometry.2 _ _ _ _ _ _ _ _ _
    he.1.2.1 hf.1.2.1 eb.2.2.1 fb.2.2.1 (component_bound _ geometry.2 e he) (component_bound _ geometry.2 f hf) addressEq
  have sequences := h.unique _ _ he.1.1 hf.1.1 blocks slots
  obtain ⟨splits,lanes⟩ := Linear.linear_injective (config m n b).width _ _ _ _ he.2 hf.2 components
  cases e with
  | mk et el => cases f with
    | mk ft fl => cases et;cases ft;simp_all

#axiomsClean source_target_separate
#axiomsClean target_injective
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KcacheCopyTriton
