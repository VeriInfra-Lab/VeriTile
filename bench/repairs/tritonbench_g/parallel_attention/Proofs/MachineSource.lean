import bench.repairs.tritonbench_g.parallel_attention.Proofs.MachineBounds
namespace VeriTile.Bench.Repairs.ParallelAttention.Machine
open VeriTile.Bench.Repairs.ParallelAttention
set_option maxHeartbeats 8000000

theorem signed32_ofNat (n : Nat) (h : Nonwrap32 n) :
    (BitVec.ofNat 32 n).toInt=(n : Int) := by
  rw [BitVec.toInt_ofNat']
  unfold Nonwrap32 at h
  apply Int.bmod_eq_of_le <;> norm_num <;> omega
theorem signed32_ofInt (i : Int) (lo : -(2^31 : Int) ≤ i) (hi : i < 2^31) :
    (BitVec.ofInt 32 i).toInt=i := by
  rw [BitVec.toInt_ofInt]
  apply Int.bmod_eq_of_le <;> norm_num <;> omega
theorem signed_add (a b : Nat) (h : Nonwrap32 (a+b)) :
    (BitVec.ofNat 32 a+BitVec.ofNat 32 b).toInt=((a+b : Nat) : Int) := by
  rw [← BitVec.ofNat_add]
  exact signed32_ofNat _ h
theorem signed_mul (a b : Nat) (h : Nonwrap32 (a*b)) :
    (BitVec.ofNat 32 a*BitVec.ofNat 32 b).toInt=((a*b : Nat) : Int) := by
  rw [← BitVec.ofNat_mul]
  exact signed32_ofNat _ h

theorem formed_intermediates (s : Shape) (g : Guard s) (width packedHead time feature : Nat)
    (hp : 0 < width) (hw : width ≤ largestWidth s) (hh : packedHead < shardHeads s)
    (ht : time < s.T+256) (hf : feature < width+128) :
    Nonwrap32 (packedHead*s.T) ∧ Nonwrap32 (packedHead*s.T*width) ∧
    Nonwrap32 (time*width) ∧ Nonwrap32 feature ∧
    Nonwrap32 (packedHead*s.T*width+time*width) ∧
    Nonwrap32 (formedOffset s width packedHead time feature) := by
  have hb := formed_machine s g width packedHead time feature hw hh ht hf
  have hm := Nat.mul_le_mul_left (packedHead*s.T) hp
  simp only [Nat.mul_one] at hm
  unfold Nonwrap32 formedOffset at *
  omega

theorem source_pointer_signed (s : Shape) (g : Guard s) (width packedHead time feature : Nat)
    (hw : width ≤ largestWidth s) (hh : packedHead < shardHeads s)
    (ht : time < s.T+256) (hf : feature < width+128) :
    (BitVec.ofNat 32 packedHead*BitVec.ofNat 32 (s.T*width)+
      BitVec.ofNat 32 time*BitVec.ofNat 32 width+BitVec.ofNat 32 feature).toInt=
      ((formedOffset s width packedHead time feature : Nat) : Int) := by
  rw [← BitVec.ofNat_mul,← BitVec.ofNat_mul,← BitVec.ofNat_add,← BitVec.ofNat_add]
  rw [source_address]
  exact signed32_ofNat _ (formed_machine s g width packedHead time feature hw hh ht hf)

theorem decoded_program (s : Shape) (h : ShapeContract s) (L x y z : Nat)
    (hx : x < (axisGrid s L).1) (hy : y < (axisGrid s L).2.1)
    (hz : z < (axisGrid s L).2.2) :
    ProgramDomain s L ⟨z,y,x/valueShards s,x%valueShards s⟩ ∧ x/valueShards s=0 := by
  have hp := shards_positive s h
  have hx' : x < valueShards s := by simpa [axisGrid,key_shards s h] using hx
  have hd := Nat.div_eq_of_lt hx'
  refine ⟨⟨hz,hy,?_,Nat.mod_lt _ hp⟩,hd⟩
  rw [hd,key_shards s h]
  change 0 < 1
  decide

theorem program_coordinates (s : Shape) (h : ShapeContract s) (g : Guard s) (a : Active s)
    (L : Nat) (hL : L=64 ∨ L=128) (tile : Tile) (valid : ProgramDomain s L tile) :
    tile.key=0 ∧ Nonwrap32 tile.bh ∧ Nonwrap32 tile.chunk ∧
      Nonwrap32 (tile.value*s.BV) ∧ Nonwrap32 (heads s*tile.value) ∧
      Nonwrap32 (tile.bh+heads s*tile.value) ∧
      Nonwrap32 (tile.bh% s.H) := by
  have hz : tile.key=0 := by have hk := valid.key; rw [key_shards s h] at hk; omega
  have hr := scalar_ranges s h g a
  have ht := block_start s.T L tile.chunk (by omega) valid.chunk
  have hc : tile.chunk ≤ tile.chunk*L := by
    simpa using Nat.mul_le_mul_left tile.chunk (show 1 ≤ L by omega)
  have hv := padded_feature s h tile.value 0 valid.value (h.value_tile_positive s)
  have hpacked := packed_head s tile.value tile.bh valid.value valid.bh
  have hwidth : s.V+127 < 2^31 := by
    have hw := largest_positive s h
    have hlim := g.formed
    have hmax := Nat.mul_le_mul_right (largestWidth s) (show 1 ≤ valueShards s*s.B*s.H*s.T+256 by omega)
    have hV := le_max_right s.K s.V
    dsimp only [formedLimit,largestWidth] at hlim
    dsimp only [largestWidth] at hmax
    omega
  have hrem := Nat.mod_lt tile.bh a.2.1
  have hH : s.H ≤ heads s := by simpa [heads] using Nat.mul_le_mul_right s.H a.1
  unfold Nonwrap32 at *
  simp only [Nat.add_zero] at hv
  exact ⟨hz,by omega,by omega,by omega,by omega,by omega,by omega⟩

/-- Inclusive j bounds also account for the state after the last advance. -/
theorem forward_scan_rows (s : Shape) (tile : Tile) (valid : ProgramDomain s 128 tile)
    (j lane : Nat) (hl : lane < 32) :
    (j ≤ tile.chunk*4 → j*32+lane < s.T+256) ∧
    (j ≤ 4 → tile.chunk*128+j*32+lane < s.T+256) := by
  constructor
  · intro hj
    exact prefix_row s 128 tile.chunk j lane (Or.inr rfl) valid.chunk (by omega) hl
  · intro hj
    exact diagonal_row s 128 tile.chunk j lane (Or.inr rfl) valid.chunk (by omega) hl
theorem backward_scan_rows (s : Shape) (tile : Tile) (valid : ProgramDomain s 64 tile)
    (j lane : Nat) (hl : lane < 32) :
    (j ≤ tile.chunk*2 → j*32+lane < s.T+256) ∧
    (j ≤ 2 → tile.chunk*64+j*32+lane < s.T+256) := by
  constructor
  · intro hj
    exact prefix_row s 64 tile.chunk j lane (Or.inl rfl) valid.chunk (by omega) hl
  · intro hj
    exact diagonal_row s 64 tile.chunk j lane (Or.inl rfl) valid.chunk (by omega) hl

/-- The descending DK/DV loop index is an Int, including its endpoint. Any
executed body has a nonnegative row and every padded lane fits the envelope. -/
theorem reverse_body_row (s : Shape) (tile : Tile) (i : Int)
    (hi : (((tile.chunk+1)*64 : Nat) : Int)-32 < i)
    (upper : i ≤ ((blocks s.T 32*32 : Nat) : Int)-32)
    (lane : Nat) (hl : lane < 32) :
    0 ≤ i ∧ i.toNat+lane < s.T+256 ∧ (i.toNat : Int)=i := by
  have hp := blocks_padding s.T 32
  have hn : 0 ≤ i := by omega
  have he := Int.toNat_of_nonneg hn
  refine ⟨hn,?_,he⟩
  omega
theorem reverse_endpoint_signed (s : Shape) (h : ShapeContract s) (g : Guard s) (a : Active s)
    (tile : Tile) (valid : ProgramDomain s 64 tile) :
    (BitVec.ofInt 32 ((((blocks s.T 32)*32 : Nat) : Int)-32)).toInt=
      (((blocks s.T 32)*32 : Nat) : Int)-32 ∧
    (BitVec.ofInt 32 ((((tile.chunk+1)*64 : Nat) : Int)-32)).toInt=
      (((tile.chunk+1)*64 : Nat) : Int)-32 := by
  have hr := (scalar_ranges s h g a).2.2.2.2.2.2.2.2
  have hp := blocks_padding s.T 32
  have ht := block_start s.T 64 tile.chunk (by decide) valid.chunk
  unfold Nonwrap32 at hr
  constructor <;> apply signed32_ofInt <;> omega
theorem local_mask_coordinate (L step lane : Nat) (hL : L=64 ∨ L=128)
    (hs : step*32 ≤ L) (hl : lane < 32) : Nonwrap32 (step*32+lane) := by
  unfold Nonwrap32
  omega

/-- The coordinate address used by each original graph load, before its
boundary check, is the same canonical dense word address as source pointers. -/
theorem graph_load_address (s : Shape) (width head time feature : Nat) :
    formedOffset s width head time feature=dense s.T width head time feature := by
  unfold formedOffset dense
  ring
theorem active_access (s : Shape) (width shards shard head time feature : Nat)
    (hs : shard < shards) (hh : head < heads s) (ht : time < s.T) (hf : feature < width) :
    formedOffset s width (head+heads s*shard) time feature < shards*heads s*s.T*width := by
  have hb : head+heads s*shard < shards*heads s := by nlinarith
  simpa [formedOffset,dense,Nat.add_mul] using dense_bound s.T width (shards*heads s) _ time feature hb ht hf

theorem empty_events (s : Shape) (h : Empty s) (scratch : Bool) (kinds : List OutputKind) :
    events s scratch kinds=[] := by
  have hc (kind : OutputKind) : kind.count s=0 := by
    have he := empty_cells s h kind
    change kind.shards s*kind.cells s=0
    rw [he,Nat.mul_zero]
  simp [events,hc,empty_cells s h]
theorem empty_combined (s : Shape) (h : Empty s) (a : Ops α) (p : Plans) (b : Bases)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat) (kinds : List OutputKind)
    (before : Memory α) :
    combinedRun a p s b trees scratch output (events s true kinds) (events s false kinds) before=before := by
  rw [empty_events s h,empty_events s h]
  rfl

#axiomsClean signed32_ofNat
#axiomsClean signed32_ofInt
#axiomsClean signed_add
#axiomsClean signed_mul
#axiomsClean formed_intermediates
#axiomsClean source_pointer_signed
#axiomsClean decoded_program
#axiomsClean program_coordinates
#axiomsClean forward_scan_rows
#axiomsClean backward_scan_rows
#axiomsClean reverse_body_row
#axiomsClean reverse_endpoint_signed
#axiomsClean graph_load_address
#axiomsClean active_access
#axiomsClean empty_events
#axiomsClean empty_combined
end VeriTile.Bench.Repairs.ParallelAttention.Machine
