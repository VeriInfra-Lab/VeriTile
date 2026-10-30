import bench.repairs.tritonbench_g.parallel_attention.Proofs.MachineSource
namespace VeriTile.Bench.Repairs.ParallelAttention.Machine
open VeriTile.Bench.Repairs.ParallelAttention
set_option maxHeartbeats 6000000

theorem future_iteration (s : Shape) (tile : Tile) (j : Nat)
    (hj : j < blocks s.T 32-(tile.chunk+1)*2) :
    let block := blocks s.T 32-1-j
    block < blocks s.T 32 ∧ (tile.chunk+1)*2 ≤ block ∧
    (((block*32 : Nat) : Int))=
      (((blocks s.T 32*32 : Nat) : Int)-32-(j : Int)*32) := by
  dsimp only
  constructor
  · omega
  constructor
  · omega
  · omega

theorem future_formed_row (s : Shape) (tile : Tile) (j lane : Nat)
    (hj : j < blocks s.T 32-(tile.chunk+1)*2) (hl : lane < 32) :
    (blocks s.T 32-1-j)*32+lane < s.T+256 := by
  have hb := (future_iteration s tile j hj).1
  have hr := small_row s _ lane hb hl
  omega

theorem future_loop_test (s : Shape) (tile : Tile) (j : Nat)
    (hj : j < blocks s.T 32-(tile.chunk+1)*2) :
    ((((tile.chunk+1)*64 : Nat) : Int)-32) <
      (((blocks s.T 32*32 : Nat) : Int)-32-(j : Int)*32) := by omega

theorem reverse_trip_count (s : Shape) (tile : Tile) (j : Nat) :
    j < blocks s.T 32-(tile.chunk+1)*2 ↔
      ((((tile.chunk+1)*64 : Nat) : Int)-32) <
        (((blocks s.T 32*32 : Nat) : Int)-32-(j : Int)*32) := by omega

theorem query_tile_addresses (s : Shape) (h : ShapeContract s) (g : Guard s)
    (L : Nat) (hL : L=64 ∨ L=128) (tile : Tile) (valid : ProgramDomain s L tile)
    (row keyLane valueLane : Nat) (hr : row < L) (hk : keyLane < s.BK) (hv : valueLane < s.BV) :
    Nonwrap32 (formedOffset s s.K tile.bh (tile.chunk*L+row) (tile.key*s.BK+keyLane)) ∧
    Nonwrap32 (formedOffset s s.V tile.bh (tile.chunk*L+row) (tile.value*s.BV+valueLane)) ∧
    Nonwrap32 ((tile.bh+heads s*tile.key)*s.T+(tile.chunk*L+row)) := by
  have ht : tile.chunk*L+row < s.T+256 := by
    have hb := padded_row s L tile.chunk row hL valid.chunk hr
    omega
  exact ⟨lt_trans (query_formed s h _ _ _ _ valid.bh ht valid.key hk) g.formed,
    lt_trans (value_formed s h _ _ _ _ valid.bh ht valid.value hv) g.formed,
    lt_trans (normalizer_formed s h _ _ _ valid.bh ht valid.key) g.formed⟩

theorem scratch_tile_addresses (s : Shape) (h : ShapeContract s) (g : Guard s)
    (L : Nat) (hL : L=64 ∨ L=128) (tile : Tile) (valid : ProgramDomain s L tile)
    (row keyLane valueLane : Nat) (hr : row < L) (hk : keyLane < s.BK) (hv : valueLane < s.BV) :
    Nonwrap32 (formedOffset s s.K (tile.bh+heads s*tile.value)
      (tile.chunk*L+row) (tile.key*s.BK+keyLane)) ∧
    Nonwrap32 (formedOffset s s.V (tile.bh+heads s*tile.key)
      (tile.chunk*L+row) (tile.value*s.BV+valueLane)) := by
  have ht : tile.chunk*L+row < s.T+256 := by
    have hb := padded_row s L tile.chunk row hL valid.chunk hr
    omega
  exact ⟨lt_trans (query_scratch_formed s h _ _ _ _ _ valid.bh ht valid.value valid.key hk) g.formed,
    lt_trans (value_scratch_formed s h _ _ _ _ _ valid.bh ht valid.value valid.key hv) g.formed⟩

/-- On each source loop, even the final pointer advance forms valid signed
indices. Select a row using forward_scan_rows or backward_scan_rows. -/
theorem loop_addresses (s : Shape) (h : ShapeContract s) (g : Guard s)
    (L : Nat) (tile : Tile) (valid : ProgramDomain s L tile)
    (time keyLane valueLane : Nat) (ht : time < s.T+256)
    (hk : keyLane < s.BK) (hv : valueLane < s.BV) :
    Nonwrap32 (formedOffset s s.K tile.bh time (tile.key*s.BK+keyLane)) ∧
    Nonwrap32 (formedOffset s s.V tile.bh time (tile.value*s.BV+valueLane)) ∧
    Nonwrap32 (tile.bh*s.T+time) := by
  have hz : tile.key=0 := by have hk := valid.key; rw [key_shards s h] at hk; omega
  exact ⟨lt_trans (query_formed s h _ _ _ _ valid.bh ht valid.key hk) g.formed,
    lt_trans (value_formed s h _ _ _ _ valid.bh ht valid.value hv) g.formed,
    by simpa [hz] using lt_trans (normalizer_formed s h _ _ _ valid.bh ht valid.key) g.formed⟩

theorem count_bound (s : Shape) (h : ShapeContract s) (kind : OutputKind) :
    kind.cells s ≤ formedLimit s ∧ kind.count s ≤ formedLimit s := by
  have hn := shards_positive s h
  have hw := largest_positive s h
  have hk := le_max_left s.K s.V
  have hv := le_max_right s.K s.V
  have hb := Nat.mul_le_mul_right (s.B*s.H*s.T) hn
  have he : valueShards s*(s.B*s.H*s.T)=valueShards s*s.B*s.H*s.T := by ring
  rw [he] at hb
  have hK := Nat.mul_le_mul_left (valueShards s*s.B*s.H*s.T) hk
  have hV := Nat.mul_le_mul_left (valueShards s*s.B*s.H*s.T) hv
  have hQ := Nat.mul_le_mul_right s.K hb
  have hW := Nat.mul_le_mul_right s.V hb
  have hZ := Nat.mul_le_mul_left (valueShards s*s.B*s.H*s.T) hw
  simp only [Nat.mul_one] at hZ
  cases kind <;> simp only [OutputKind.cells,OutputKind.count,OutputKind.width,OutputKind.shards,Nat.one_mul,Nat.mul_one]
  all_goals
    dsimp only [formedLimit,largestWidth,valueShards] at *
    change _ ≤ _ ∧ _ ≤ _
    constructor <;> nlinarith

theorem host_cell_machine (s : Shape) (h : ShapeContract s) (g : Guard s)
    (kind : OutputKind) (i : Nat) (hi : i < kind.cells s) : Nonwrap32 i :=
  lt_trans (lt_of_lt_of_le hi (count_bound s h kind).1) g.formed
theorem scratch_cell_machine (s : Shape) (h : ShapeContract s) (g : Guard s)
    (kind : OutputKind) (i : Nat) (hi : i < kind.count s) : Nonwrap32 i :=
  lt_trans (lt_of_lt_of_le hi (count_bound s h kind).2) g.formed

theorem cdiv_operands (s : Shape) (h : ShapeContract s) (g : Guard s) (a : Active s) :
    Nonwrap32 (s.V+s.BV-1) ∧ Nonwrap32 (s.K+s.BK-1) ∧
    Nonwrap32 (s.T+127) ∧ Nonwrap32 s.B ∧ Nonwrap32 s.H := by
  have cap := tile_caps s h
  have ranges := scalar_ranges s h g a
  have hV : s.V+127 < 2^31 := by
    have hlim := g.formed
    have hmax := Nat.mul_le_mul_right (largestWidth s)
      (show 1 ≤ valueShards s*s.B*s.H*s.T+256 by omega)
    have hv := le_max_right s.K s.V
    dsimp only [formedLimit,largestWidth] at hlim
    dsimp only [largestWidth] at hmax
    omega
  have hB : s.B ≤ heads s := by simpa [heads] using Nat.mul_le_mul_left s.B a.2.1
  have hH : s.H ≤ heads s := by simpa [heads] using Nat.mul_le_mul_right s.H a.1
  have hk := h.key_limit
  unfold Nonwrap32 at *
  omega

theorem control_operands (s : Shape) (h : ShapeContract s) (g : Guard s) (a : Active s)
    (L : Nat) (hL : L=64 ∨ L=128) (tile : Tile) (valid : ProgramDomain s L tile) :
    Nonwrap32 (tile.chunk*L) ∧ Nonwrap32 ((tile.chunk+1)*L) ∧
    Nonwrap32 (blocks s.T 32*32) ∧ 0 < s.H := by
  have ranges := scalar_ranges s h g a
  have hb := block_start s.T L tile.chunk (by omega) valid.chunk
  have hp := blocks_padding s.T 32
  have hn := a.2.1
  unfold Nonwrap32 at *
  rcases hL with rfl|rfl <;> omega

theorem pointer_bits_exact (s : Shape) (bases : List Nat) (abi : PointerABI s bases)
    (base : Nat) (hb : base∈bases) (offset : Nat) (ho : offset < formedLimit s) :
    (BitVec.ofNat 64 (4*(base+offset))).toNat=4*(base+offset) := by
  have hp := pointer_safe s bases abi base hb offset ho
  rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt]
  omega

#axiomsClean future_iteration
#axiomsClean future_formed_row
#axiomsClean reverse_trip_count
#axiomsClean query_tile_addresses
#axiomsClean scratch_tile_addresses
#axiomsClean loop_addresses
#axiomsClean count_bound
#axiomsClean host_cell_machine
#axiomsClean scratch_cell_machine
#axiomsClean cdiv_operands
#axiomsClean control_operands
#axiomsClean pointer_bits_exact
end VeriTile.Bench.Repairs.ParallelAttention.Machine
