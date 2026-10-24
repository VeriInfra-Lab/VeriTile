import bench.repairs.tritonbench_g.kv_cache_copy.Proofs.Contract
import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.Memory

namespace VeriTile.Bench.Repairs.KvCacheCopy
noncomputable section
set_option maxHeartbeats 4000000
namespace Linear
export VeriTile.Bench.Repairs.RowwiseQuantizationTriton (linear_bound linear_injective)
end Linear

def ValidTile (p : Config) (e : Tile) : Prop := e.sequence<p.count ∧ e.head<p.heads ∧ e.split<p.splits
def ValidLane (p : Config) (e : Lane) : Prop := ValidTile p e.tile ∧ e.lane<p.width

theorem tiles_member (p : Config) (e : Tile) : e∈tiles p ↔ ValidTile p e := by
  cases e
  simp [tiles,ValidTile]

theorem tileLanes_member (p : Config) (t : Tile) (e : Lane) :
    e∈tileLanes p t ↔ e.tile=t ∧ e.lane<p.width := by
  cases e with
  | mk tile key lane => cases key <;> simp [tileLanes,eq_comm,and_comm]

theorem lanes_member (p : Config) (e : Lane) : e∈lanes p ↔ ValidLane p e := by
  simp only [lanes,List.mem_flatMap,tiles_member,tileLanes_member,ValidLane]
  constructor
  · rintro ⟨t,ht,he,hc⟩; exact ⟨he ▸ ht,hc⟩
  · rintro ⟨ht,hc⟩; exact ⟨e.tile,ht,rfl,hc⟩

theorem component_bound (p : Config) (dims : p.splits*p.width=p.dims) (e : Lane) (he : ValidLane p e) :
    component p e<p.dims := by
  rw [←dims]
  exact Linear.linear_bound _ _ _ _ he.1.2.2 he.2

theorem cache_bound (p : Config) (width : 0<p.width) (dims : p.splits*p.width=p.dims)
    (mode : Bool) (block h position d : Nat) (hb : block<p.blocks) (hh : h<p.heads)
    (hp : position<p.blockSize) (hd : d<p.dims) :
    cacheIndex p mode block h position d<p.blocks*p.heads*p.blockSize*p.dims := by
  have bh := Linear.linear_bound p.blocks p.heads block h hb hh
  cases mode with
  | «false» => exact Linear.linear_bound _ _ _ d (Linear.linear_bound _ _ _ position bh hp) hd
  | «true» =>
    have split : d/p.width<p.splits := (Nat.div_lt_iff_lt_mul width).mpr (by simpa [dims] using hd)
    have lane : d%p.width<p.width := Nat.mod_lt _ width
    have b := Linear.linear_bound _ _ _ _
      (Linear.linear_bound _ _ _ position (Linear.linear_bound _ _ _ _ bh split) hp) lane
    change (((block*p.heads+h)*p.splits+d/p.width)*p.blockSize+position)*p.width+d%p.width<_
    have size : p.blocks*p.heads*p.splits*p.blockSize*p.width=p.blocks*p.heads*p.blockSize*p.dims := by rw [←dims];ring
    exact size ▸ b

theorem cache_injective (p : Config) (width : 0<p.width) (dims : p.splits*p.width=p.dims)
    (mode : Bool) (b c h k s t d e : Nat) (hh : h<p.heads) (hk : k<p.heads)
    (hs : s<p.blockSize) (ht : t<p.blockSize) (hd : d<p.dims) (he : e<p.dims)
    (same : cacheIndex p mode b h s d=cacheIndex p mode c k t e) : b=c ∧ h=k ∧ s=t ∧ d=e := by
  cases mode with
  | «false» =>
    obtain ⟨outer,de⟩ := Linear.linear_injective p.dims _ _ d e hd he same
    obtain ⟨outer,st⟩ := Linear.linear_injective p.blockSize _ _ s t hs ht outer
    obtain ⟨bc,hk⟩ := Linear.linear_injective p.heads b c h k hh hk outer
    exact ⟨bc,hk,st,de⟩
  | «true» =>
    have hd' : d/p.width<p.splits := (Nat.div_lt_iff_lt_mul width).mpr (by simpa [dims] using hd)
    have he' : e/p.width<p.splits := (Nat.div_lt_iff_lt_mul width).mpr (by simpa [dims] using he)
    obtain ⟨outer,lanes⟩ := Linear.linear_injective p.width _ _ _ _ (Nat.mod_lt d width) (Nat.mod_lt e width) same
    obtain ⟨outer,st⟩ := Linear.linear_injective p.blockSize _ _ s t hs ht outer
    obtain ⟨outer,parts⟩ := Linear.linear_injective p.splits _ _ _ _ hd' he' outer
    obtain ⟨bc,hk⟩ := Linear.linear_injective p.heads b c h k hh hk outer
    refine ⟨bc,hk,st,?_⟩
    have da := Nat.div_add_mod d p.width
    have ea := Nat.div_add_mod e p.width
    rw [parts,lanes] at da
    omega

def Inside (m : Inputs) (r : Role) (a : Nat) : Prop := base m r≤a ∧ a<base m r+4*(m r).numel

theorem in_address (m : Inputs) (r : Role) (i : Nat) (hi : i<(m r).numel) : Inside m r (address m r i) := by
  unfold Inside address
  omega

theorem source_bound (m : Inputs) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m b l t)
    (e : Lane) (he : ValidLane (config m b) e) :
    Inside m (inputRole e.key) (sourceAddress m (config m b) e) := by
  apply in_address
  have dims := component_bound (config m b) (split_geometry m b l t h).2 e he
  have flat := Linear.linear_bound _ _ _ _
    (Linear.linear_bound _ _ _ _ he.1.1 he.1.2.1) dims
  have sizes := numel m b l t h
  cases hk : e.key <;> simpa only [inputRole,hk,Bool.false_eq_true,ite_false,ite_true,inputIndex,
    sizes.1,sizes.2.1] using flat

theorem target_bound (m : Inputs) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m b l t)
    (e : Lane) (he : ValidLane (config m b) e) :
    Inside m (outputRole e.key) (targetAddress m (config m b) l t e) := by
  apply in_address
  have geometry := split_geometry m b l t h
  have bounds := metadata_bounds m b l t h e.tile.sequence he.1.1
  have flat := cache_bound (config m b) geometry.1 geometry.2 (e.key && (config m b).newLayout)
    _ _ _ _ bounds.2.2.2 he.1.2.1 bounds.2.2.1 (component_bound _ geometry.2 e he)
  have sizes := numel m b l t h
  cases hk : e.key <;> simpa only [outputRole,hk,Bool.false_eq_true,ite_false,ite_true,outputIndex,
    sizes.2.2.1,sizes.2.2.2.1] using flat

theorem separate_inside (m : Inputs) (separate : Separate m) (r s : Role) (different : r≠s)
    (a : Nat) (hr : Inside m r a) (hs : Inside m s a) : False := by
  rcases separate r s different with before|after <;> unfold Inside at hr hs <;> omega

theorem source_target_separate (m : Inputs) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m b l t)
    (e f : Lane) (he : ValidLane (config m b) e) (hf : ValidLane (config m b) f) :
    sourceAddress m (config m b) e≠targetAddress m (config m b) l t f := by
  intro same
  have source := source_bound m b l t h e he
  rw [same] at source
  exact separate_inside m h.separate _ _ (by cases e.key <;> cases f.key <;> decide) _
    source (target_bound m b l t h f hf)

theorem target_injective (m : Inputs) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m b l t)
    (e f : Lane) (he : ValidLane (config m b) e) (hf : ValidLane (config m b) f)
    (same : targetAddress m (config m b) l t e=targetAddress m (config m b) l t f) : e=f := by
  have key : e.key=f.key := by
    by_contra different
    have er := target_bound m b l t h e he
    rw [same] at er
    exact separate_inside m h.separate _ _ (by cases heq : e.key <;> cases hfq : f.key <;> simp_all [outputRole]) _
      er (target_bound m b l t h f hf)
  have addressEq : outputIndex (config m b) l t e.key e.tile.sequence e.tile.head (component (config m b) e)=
      outputIndex (config m b) l t f.key f.tile.sequence f.tile.head (component (config m b) f) := by
    unfold targetAddress address at same
    rw [key] at same ⊢
    omega
  unfold outputIndex at addressEq
  rw [key] at addressEq
  have geometry := split_geometry m b l t h
  have eb := metadata_bounds m b l t h e.tile.sequence he.1.1
  have fb := metadata_bounds m b l t h f.tile.sequence hf.1.1
  obtain ⟨blocks,heads,slots,components⟩ := cache_injective (config m b) geometry.1 geometry.2 _ _ _ _ _ _ _ _ _
    he.1.2.1 hf.1.2.1 eb.2.2.1 fb.2.2.1 (component_bound _ geometry.2 e he) (component_bound _ geometry.2 f hf) addressEq
  have sequences := h.unique _ _ he.1.1 hf.1.1 blocks slots
  obtain ⟨splits,lanes⟩ := Linear.linear_injective (config m b).width _ _ _ _ he.2 hf.2 components
  cases e with
  | mk et ek el => cases f with
    | mk ft fk fl => cases et; cases ft; simp_all

#axiomsClean cache_injective
#axiomsClean source_target_separate
#axiomsClean target_injective
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KvCacheCopy
