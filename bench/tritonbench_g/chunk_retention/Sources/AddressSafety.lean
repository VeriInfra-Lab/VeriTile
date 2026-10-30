import bench.tritonbench_g.chunk_retention.Sources.Geometry

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001
noncomputable section

namespace Shape

theorem block_cover (size tile : Nat) (ht : 0<tile) :
    size≤((size+tile-1)/tile)*tile := by
  have h := Nat.mod_lt (size+tile-1) ht
  have e := Nat.div_add_mod (size+tile-1) tile
  rw [Nat.mul_comm tile] at e
  omega

theorem key_cover (c : Shape) : c.K≤c.paddedKey :=
  block_cover c.K c.BK c.key_tile_pos

theorem value_cover (c : Shape) : c.V≤c.paddedValue :=
  block_cover c.V c.BV c.value_tile_pos

theorem feature_block (size tile index : Nat) (ht : 0<tile) (hi : index<size) :
    index/tile<(size+tile-1)/tile := by
  have hc := block_cover size tile ht
  exact (Nat.div_lt_iff_lt_mul ht).mpr (lt_of_lt_of_le hi hc)

theorem key_offset_bound (c : Shape) (head row key : Nat)
    (hh : head<c.heads) (hr : row<c.T) (hk : key<c.K) :
    c.keyOffset head row key<c.keySize := by
  have h := digit_bound head c.heads (row*c.K+key) (c.T*c.K) hh (digit_bound row c.T key c.K hr hk)
  simpa [keyOffset,keyHead,keySize,Nat.mul_assoc,Nat.add_assoc] using h

theorem value_offset_bound (c : Shape) (head row value : Nat)
    (hh : head<c.heads) (hr : row<c.T) (hv : value<c.V) :
    c.valueOffset head row value<c.valueSize := by
  have h := digit_bound head c.heads (row*c.V+value) (c.T*c.V) hh (digit_bound row c.T value c.V hr hv)
  simpa [valueOffset,valueHead,valueSize,Nat.mul_assoc,Nat.add_assoc] using h

theorem state_offset_bound (c : Shape) (head key value : Nat)
    (hh : head<c.heads) (hk : key<c.K) (hv : value<c.V) :
    c.stateOffset head key value<c.stateSize := by
  have h := digit_bound head c.heads (key*c.V+value) (c.K*c.V) hh (digit_bound key c.K value c.V hk hv)
  simpa [stateOffset,stateHead,stateSize,Nat.mul_assoc,Nat.add_assoc] using h

theorem history_offset_bound (c : Shape) (head chunk key value : Nat)
    (hh : head<c.heads) (hc : chunk<c.chunks) (hk : key<c.K) (hv : value<c.V) :
    c.historyOffset head chunk key value<c.historySize := by
  have inner := digit_bound chunk c.chunks (key*c.V+value) (c.K*c.V) hc (digit_bound key c.K value c.V hk hv)
  have h := digit_bound head c.heads (chunk*(c.K*c.V)+(key*c.V+value))
    (c.chunks*(c.K*c.V)) hh inner
  simpa [historyOffset,historyHead,stateHead,historySize,Nat.mul_assoc,Nat.add_assoc] using h

theorem partial_offset_bound (c : Shape) (block head row value : Nat)
    (hb : block<c.keyBlocks) (hh : head<c.heads) (hr : row<c.T) (hv : value<c.V) :
    c.partialOffset block head row value<c.partialSize :=
  digit_bound block c.keyBlocks (c.valueOffset head row value) c.valueSize hb
    (c.value_offset_bound head row value hh hr hv)

theorem key_offset_injective (c : Shape) (head row key head' row' key' : Nat)
    (hr : row<c.T) (hr' : row'<c.T) (hk : key<c.K) (hk' : key'<c.K)
    (he : c.keyOffset head row key=c.keyOffset head' row' key') :
    head=head' ∧ row=row' ∧ key=key' := by
  have h := digit_injective c.K (head*c.T+row) key (head'*c.T+row') key' hk hk'
    (by simpa [keyOffset,keyHead,Nat.add_mul,Nat.mul_assoc] using he)
  have hh := digit_injective c.T head row head' row' hr hr' h.1
  exact ⟨hh.1,hh.2,h.2⟩

theorem history_offset_injective (c : Shape) (head chunk key value head' chunk' key' value' : Nat)
    (hc : chunk<c.chunks) (hc' : chunk'<c.chunks)
    (hk : key<c.K) (hk' : key'<c.K) (hv : value<c.V) (hv' : value'<c.V)
    (he : c.historyOffset head chunk key value=c.historyOffset head' chunk' key' value') :
    head=head' ∧ chunk=chunk' ∧ key=key' ∧ value=value' := by
  have h := digit_injective c.V ((head*c.chunks+chunk)*c.K+key) value
    ((head'*c.chunks+chunk')*c.K+key') value' hv hv'
    (by simpa [historyOffset,historyHead,stateHead,Nat.add_mul,Nat.mul_assoc] using he)
  have h' := digit_injective c.K (head*c.chunks+chunk) key (head'*c.chunks+chunk') key' hk hk' h.1
  have h'' := digit_injective c.chunks head chunk head' chunk' hc hc' h'.1
  exact ⟨h''.1,h''.2,h'.2,h.2⟩

/-- Every logical key element has one grid/tile decomposition. -/
theorem key_grid_cover (c : Shape) (row key : Nat) (hr : row<c.T) (hk : key<c.K) :
    row/64<c.chunks ∧ row%64<64 ∧ key/c.BK<c.keyBlocks ∧ key%c.BK<c.BK ∧
    row/64*64+row%64=row ∧ key/c.BK*c.BK+key%c.BK=key := by
  exact ⟨c.chunk_of_row row hr,Nat.mod_lt row (by decide),
    feature_block c.K c.BK key c.key_tile_pos hk,Nat.mod_lt key c.key_tile_pos,
    by simpa [Nat.mul_comm] using Nat.div_add_mod row 64,
    by simpa [Nat.mul_comm] using Nat.div_add_mod key c.BK⟩

theorem padded_lane_bound (blocks tile block lane : Nat) (hb : block<blocks) (hl : lane<tile) :
    block*tile+lane<blocks*tile := digit_bound block blocks lane tile hb hl

theorem signed_decay_range (length lane : Nat) (hl : length≤64) (hi : lane<64) :
    -(64:Int)≤(length:Int)-lane-1 ∧ (length:Int)-lane-1<64 := by omega

/-- Chunk-local K masks rule out adjacent history chunks, even in the last key tile. -/
theorem padded_history_lane_inactive (c : Shape) (key : Nat) (hk : c.K≤key) :
    ¬key<c.K := by omega
end Shape

#axiomsClean Shape.history_offset_bound
#axiomsClean Shape.history_offset_injective
#axiomsClean Shape.key_grid_cover
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001
