import bench.repairs.tritonbench_g.kcache_copy_triton.Proofs.Native
import bench.repairs.tritonbench_g.kv_cache_copy.Proofs.Source
namespace VeriTile.Bench.Repairs.KcacheCopyTriton
noncomputable section
open Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 100000

theorem tile_readonly (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m n b l t)
    (heap : Memory α) (x : Tile) (hx : ValidTile (config m n b) x)
    (r : Role) (readonly : r≠.cache) (a : Nat) (inside : Inside m r a) :
    tile m (config m n b) l t heap x a=heap a := by
  apply scatter_frame
  intro e he same
  obtain ⟨eq,hc⟩ := (tileLanes_member (config m n b) x e).mp he
  have target := target_bound m n b l t h e ⟨eq ▸ hx,hc⟩
  rw [←same] at target
  exact separate_inside m h.separate r .cache readonly a inside target

theorem tile_binding (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m n b l t)
    (decode : α → Word) (heap : Memory α) (binding : TensorBinding m n b l t decode heap)
    (x : Tile) (hx : ValidTile (config m n b) x) :
    TensorBinding m n b l t decode (tile m (config m n b) l t heap x) := by
  have sizes := numel m n b l t h
  refine ⟨binding.capacity,?_,?_⟩
  · intro s hs
    rw [tile_readonly m n b l t h heap x hx .lengths (by decide) _
      (in_address m .lengths s (by rw [sizes.2.2.1];exact hs))]
    exact binding.lengths s hs
  · intro s c hs hc
    rw [tile_readonly m n b l t h heap x hx .table (by decide) _
      (in_address m .table _ (by rw [sizes.2.2.2];exact Linear.linear_bound _ _ _ _ hs hc))]
    exact binding.table s c hs hc

namespace Native

theorem tile_exact (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m n b l t)
    (decode : α → Word) (heap : Memory α) (binding : TensorBinding m n b l t decode heap)
    (x : Tile) (hx : ValidTile (config m n b) x) :
    tile m (config m n b) decode heap x=KcacheCopyTriton.tile m (config m n b) l t heap x := by
  unfold tile
  rw [resolve_exact m n b l t h decode heap binding x.sequence hx.1]
  apply KvCacheCopy.scatter_congr
  · intro e he
    obtain ⟨eq,hc⟩ := (tileLanes_member (config m n b) x e).mp he
    have pointer := target_pointer m n b l t h decode heap binding e ⟨eq ▸ hx,hc⟩
    simpa only [eq] using pointer
  · intro e he
    obtain ⟨eq,hc⟩ := (tileLanes_member (config m n b) x e).mp he
    rw [source_pointer m n b l t h decode heap binding e ⟨eq ▸ hx,hc⟩]

theorem launch_exact (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables) (h : InputContract m n b l t)
    (decode : α → Word) (heap : Memory α) (binding : TensorBinding m n b l t decode heap) :
    launch m (config m n b) decode heap=KcacheCopyTriton.launch m (config m n b) l t heap :=
  KvCacheCopy.foldl_equiv _ _ (TensorBinding m n b l t decode) (ValidTile (config m n b))
    (fun state hb x hx => tile_exact m n b l t h decode state hb x hx)
    (fun state hb x hx => tile_binding m n b l t h decode state hb x hx)
    (tiles (config m n b)) (fun x hx => (tiles_member _ _).mp hx) heap binding

#axiomsClean tile_exact
#axiomsClean launch_exact
end Native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KcacheCopyTriton
