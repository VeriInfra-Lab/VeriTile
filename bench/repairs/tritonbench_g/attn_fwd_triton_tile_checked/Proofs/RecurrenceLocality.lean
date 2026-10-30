import bench.repairs.tritonbench_g.attn_fwd_triton_tile_checked.Proofs.CursorGeometry

namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.ReadLocality
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open AttnFwdTriton.TypedExecution Graph Model
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

theorem update_eq (p : Config) (BD : Nat) (idx : ProgramIndex p)
    (left right : Memory) (same : Agree p left right) (n : Nat)
    (active : n*64<p.context) (causal : Bool) (old : Carry BD) :
    let f := fixed p BD left idx.1.val idx.2.val
    update p BD (reads left) f (n*64) (cursor f n) causal old =
      update p BD (reads right) f (n*64) (cursor f n) causal old := by
  let f := fixed p BD left idx.1.val idx.2.val
  let ptr := cursor f n
  obtain ⟨hk,hs,hv⟩ := iteration_reads p BD idx left right same n active
  have key : keyTile p BD (reads left) f (n*64) ptr =
      keyTile p BD (reads right) f (n*64) ptr := by
    funext i
    unfold keyTile
    split
    · rename_i selected
      have hi : f.keys (i.2.1,PUnit.unit)<f.context-n*64 ∧ i.1.val<p.headDim := by
        simpa using selected
      exact hk i hi.1 hi.2
    · rfl
  have value : valueTile p BD (reads left) f (n*64) ptr =
      valueTile p BD (reads right) f (n*64) ptr := by
    funext i
    unfold valueTile
    split
    · rename_i selected
      have hi : f.keys (i.1,PUnit.unit)<f.context-n*64 ∧ i.2.1.val<p.headDim := by
        simpa using selected
      exact hv i hi.1 hi.2
    · rfl
  have score : scores p BD (reads left) f (n*64) ptr causal =
      scores p BD (reads right) f (n*64) ptr causal := by
    unfold scores
    rw [key,hs]
  change update p BD (reads left) f (n*64) ptr causal old = _
  simp only [update,score,value]
  rfl

theorem fold_eq (p : Config) (BD : Nat) (idx : ProgramIndex p)
    (left right : Memory) (same : Agree p left right) (stop : Nat)
    (bounded : stop≤p.context) (causal : Bool) (n : Nat) (old : Carry BD) :
    let f := fixed p BD left idx.1.val idx.2.val
    fold p BD (reads left) f stop causal (n*64) (cursor f n) old =
      fold p BD (reads right) f stop causal (n*64) (cursor f n) old := by
  dsimp only
  conv_lhs => rw [fold]
  conv_rhs => rw [fold]
  by_cases active : n*64<stop
  · simp only [active,dif_pos]
    rw [update_eq p BD idx left right same n (lt_of_lt_of_le active bounded) causal old,
      advance_cursor,show n*64+64=(n+1)*64 from by omega]
    exact fold_eq p BD idx left right same stop bounded causal (n+1) _
  · simp [active]
termination_by stop-n*64
decreasing_by omega

theorem result_eq (p : Config) (BD : Nat) (idx : ProgramIndex p)
    (left right : Memory) (same : Agree p left right) :
    result p BD left idx.1.val idx.2.val = result p BD right idx.1.val idx.2.val := by
  have hf := fixed_eq p BD idx left right same
  unfold result
  rw [←hf]
  simp only [lower_chunk,start_cursor]
  rw [fold_eq p BD idx left right same _ (stop_bounded p BD left idx Bool.false)]
  exact fold_eq p BD idx left right same _ (stop_bounded p BD left idx Bool.true) _ _ _

theorem reference_eq (p : Config) (BD : Nat) (hD : p.headDim≤BD)
    (left right : Memory) (same : Agree p left right) (c : Coordinate p) :
    Model.reference p BD hD left c = Model.reference p BD hD right c := by
  unfold Model.reference
  rw [result_eq p BD (owner p c) left right same]

#axiomsClean result_eq
#axiomsClean reference_eq
end
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.ReadLocality
