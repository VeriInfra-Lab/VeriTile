import bench.tritonbench_g.parallel_retention_attention.Proofs.Grid

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention
noncomputable section

/-- Closed form of min(128,max(16,next_power_of_2(D))) for positive D. -/
def featureWidth (D : Nat) : Nat :=
  if D ≤ 16 then 16 else if D ≤ 32 then 32 else if D ≤ 64 then 64 else 128

theorem featureWidth_tiles (D : Nat) :
    featureWidth D=16 ∨ featureWidth D=32 ∨ featureWidth D=64 ∨ featureWidth D=128 := by
  unfold featureWidth
  split_ifs <;> simp_all

theorem featureWidth_positive (D : Nat) : 0<featureWidth D := by
  rcases featureWidth_tiles D with h | h | h | h <;> rw [h] <;> norm_num

theorem blocks_upper (n b : Nat) (hb : 0<b) : blocks n b*b<n+b := by
  have h := Nat.div_mul_le_self (n+b-1) b
  unfold blocks
  omega

/-- Includes offset vectors for inactive lanes and post-loop pointer advances;
these are obligations before the source's explicit signed32 offset casts. -/
theorem source_time_coordinates (s : Shape) (domain : IndexDomain s.B s.H s.T s.DK s.DV s.BK s.BV)
    (L c r : Nat) (tile : L=32 ∨ L=64 ∨ L=128) (hc : c<blocks s.T L) (hr : r ≤ L) :
    c*L+r<2^31 := by
  have hL : 0<L := by rcases tile with h | h | h <;> omega
  have hmax : L ≤ 128 := by rcases tile with h | h | h <;> omega
  have cover := blocks_cover s.T 128 (by norm_num)
  have upper := blocks_upper s.T L hL
  have hc' : c+1 ≤ blocks s.T L := by omega
  have hb : c*L+r ≤ blocks s.T L*L := by nlinarith
  have hd := domain.time_padded
  omega

theorem source_key_coordinates (s : Shape) (domain : IndexDomain s.B s.H s.T s.DK s.DV s.BK s.BV)
    (k e : Nat) (hk : k<blocks s.DK s.BK) (he : e ≤ s.BK) : k*s.BK+e<2^31 := by
  have h := domain.key_padded
  have hk' : k+1 ≤ blocks s.DK s.BK := by omega
  nlinarith

theorem source_value_coordinates (s : Shape) (domain : IndexDomain s.B s.H s.T s.DK s.DV s.BK s.BV)
    (v e : Nat) (hv : v<blocks s.DV s.BV) (he : e ≤ s.BV) : v*s.BV+e<2^31 := by
  have h := domain.value_padded
  have hv' : v+1 ≤ blocks s.DV s.BV := by omega
  nlinarith

/-- Signed diagonal subtraction is formed without truncated Nat subtraction. -/
theorem signed_diagonal (L row col : Nat) (hL : L ≤ 128) (hr : row<L) (hc : col<L) :
    -(2:ℤ)^31 ≤ (row:ℤ)-(col:ℤ) ∧ (row:ℤ)-(col:ℤ)<2^31 := by
  omega

structure PointerDomain (byteBase formedCells : Nat) : Prop where
  aligned : byteBase%4=0
  no_wrap : byteBase+formedCells*4<2^64

theorem formed_address (byteBase maxCells cell : Nat) (domain : PointerDomain byteBase maxCells)
    (bound : cell ≤ maxCells) : byteBase+cell*4<2^64 := by
  have h := domain.no_wrap
  omega

theorem signed64_byte_offsets (maxCells cell : Nat) (hmax : maxCells<2^61) (h : cell ≤ maxCells) :
    cell<2^63 ∧ cell*4<2^63 := by omega

/-- No zero grid is launched, and positive feature dimensions retain at least
one shard even when B, H or T is zero. -/
theorem blocks_positive (n b : Nat) (hn : 0<n) (hb : 0<b) : 0<blocks n b := by
  have h := blocks_cover n b hb
  by_contra hz
  have he : blocks n b=0 := by omega
  rw [he] at h
  omega

theorem empty_counts (s : Shape) (kind : OutputKind) (empty : s.B*s.H*s.T=0) :
    kind.cells s=0 ∧ kind.count s=0 := by
  simp [OutputKind.cells,OutputKind.count,empty]

#axiomsClean featureWidth_tiles
#axiomsClean source_time_coordinates
#axiomsClean source_key_coordinates
#axiomsClean source_value_coordinates
#axiomsClean signed_diagonal
#axiomsClean formed_address
#axiomsClean signed64_byte_offsets
#axiomsClean blocks_positive
#axiomsClean empty_counts
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention
