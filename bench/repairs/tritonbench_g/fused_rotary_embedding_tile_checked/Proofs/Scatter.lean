import bench.repairs.tritonbench_g.fused_rotary_embedding_tile_checked.Proofs.Program
import bench.repairs.tritonbench_g.fused_rotary_embedding_tile_checked.Proofs.Layout

namespace VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.Source
open VeriTile.Triton FP.Structural FusedRotaryEmbedding
noncomputable section
set_option maxHeartbeats 2000000
variable {α : Type}

theorem scatter_congr {ι : Type} (xs : List ι) (r : RegionName)
    (mask : ι → Prop) [DecidablePred mask] (off off' : ι → Nat) (value value' : ι → Cell α)
    (offsets : ∀ i∈xs,mask i → off i=off' i)
    (values : ∀ i∈xs,mask i → value i=value' i) (s : State α) :
    scatter xs r mask off value s=scatter xs r mask off' value' s := by
  induction xs generalizing s with
  | nil=>rfl
  | cons i xs ih=>
    simp only [scatter,List.foldl_cons]
    by_cases hi : mask i
    · simp only [hi,ite_true,offsets i (by simp) hi,values i (by simp) hi]
      exact ih (fun j hj=>offsets j (by simp [hj])) (fun j hj=>values j (by simp [hj])) _
    · simp only [hi,ite_false]
      exact ih (fun j hj=>offsets j (by simp [hj])) (fun j hj=>values j (by simp [hj])) _

theorem scatter_lift (c : Config) (m : Memory α) (kind : Kind) (n limit : Nat) (bound : limit≤n)
    (mask : TileIndex [n] → Prop) [DecidablePred mask]
    (off : TileIndex [n] → Nat) (value : TileIndex [n] → Cell α)
    (naturalOff : Nat → Nat) (naturalValue : Nat → α)
    (masks : ∀ i,mask i↔i.1.val<limit)
    (offsets : ∀ i,mask i → off i=naturalOff i.1.val)
    (values : ∀ i,mask i → value i=.mk .real (naturalValue i.1.val))
    (s : State α) (memory : s.mem=lift c m) :
    (scatter (TileShape.allIndices [n]) (region kind.output) mask off value s).mem=
      lift c (Stores.commit ((List.range limit).map (fun i=>((kind.output,naturalOff i),naturalValue i))) m) := by
  rw [scatter_congr _ _ _ off (fun i=>naturalOff i.1.val) value
    (fun i=>.mk .real (naturalValue i.1.val)) (fun i _=>offsets i) (fun i _=>values i)]
  unfold scatter
  rw [State.masked_scatter_eq_filter,lift_commit c m kind _ _ _ s memory]
  have masks' : (fun i : TileIndex [n]=>decide (mask i))=(fun i=>decide (i.1.val<limit)) := by
    funext i
    simp [masks i]
  rw [masks',active_lanes n limit bound (fun i=>((kind.output,naturalOff i),naturalValue i))]

#axiomsClean scatter_lift
end
end VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.Source
