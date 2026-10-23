import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.Bounds

namespace VeriTile.Bench.Repairs.CacheTransform
set_option maxRecDepth 10000

/-- Complete native address workset for one source tile. Extents are supplied
by successful ordinary allocations, while all pointer/intermediate bounds
are proved. The masked tail is checked, including its unmasked metadata load. -/
theorem native_tile_safety (n width capacity k lane : Nat) (rows : Nat → Nat)
    (hcount : n * width ≤ i64Max) (hcache : capacity * width ≤ i64Max)
    (hmetadata : n * 8 ≤ i64Max) (hcapacity : capacity ≤ i64Max)
    (hrows : ∀ q, q < n → rows q < capacity)
    (hk : k < blocks (n * width)) (hl : lane < 256) :
    let idx := k * 256 + lane
    let safe := if idx < n * width then idx else 0
    let q := safe / width
    let offset := rows q * width + safe % width
    0 < width ∧ idx < u64Mod ∧ safe < n * width ∧
    q < n ∧ q * 8 + 7 < n * 8 ∧ q * 8 + 7 < 9223372036854775808 ∧
    rows q < 9223372036854775808 ∧ rows q * width < u64Mod ∧
    offset < capacity * width ∧ offset < 9223372036854775808 ∧
    safe < 9223372036854775808 ∧ localPid k < 2147483647 := by
  dsimp
  have hnonempty : 0 < n * width := lt_of_le_of_lt (Nat.zero_le _) (block_base_bound _ _ hk)
  have hw : 0 < width := Nat.pos_of_mul_pos_left hnonempty
  have hi := index_u64_bound (n * width) k lane hcount hk hl
  have hs := safe_bounds (n * width) (k * 256 + lane) hnonempty
  have hb := gather_bounds n width capacity _ rows hw hnonempty hs hrows
  have hm := metadata_byte_bound n _ 7 hb.1 (by decide)
  have hma := address_i64 _ _ hm hmetadata
  have hca := address_i64 _ _ hb.2.2 hcache
  have hsa := address_i64 _ _ hs hcount
  have hra : rows ((if k * 256 + lane < n * width then k * 256 + lane else 0) / width) <
      9223372036854775808 := address_i64 _ _ (hrows _ hb.1) hcapacity
  refine ⟨hw, hi, hs, hb.1, hm, hma, hra, ?_, hb.2.2, hca, hsa, localPid_bound k⟩
  unfold u64Mod
  omega

/-- Bounds imply equality with native unsigned wrap semantics at every
indexing addition/product; this is an integer fact, not a numerical port. -/
theorem unsigned_wrap_identity (value : Nat) (h : value < u64Mod) :
    value % 18446744073709551616 = value := Nat.mod_eq_of_lt h

theorem signed_row_cast_identity (row : Nat) (h : row < 9223372036854775808) :
    ((row : Int) % 18446744073709551616).toNat = row := by
  rw [Int.emod_eq_of_lt (by omega) (by omega), Int.toNat_natCast]

#axiomsClean native_tile_safety
#axiomsClean signed_row_cast_identity
end VeriTile.Bench.Repairs.CacheTransform
