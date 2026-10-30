import bench.tritonbench_g.cache_transform_triton.Proofs.Math

namespace VeriTile.Bench.Repairs.CacheTransform
open VeriTile.Triton
set_option maxRecDepth 5000

def i64Max : Nat := 9223372036854775807
def u64Mod : Nat := 18446744073709551616
def blockSize : Nat := 256
def gridLimit : Nat := 2147483647
def blocks (count : Nat) : Nat := (count + 255) / 256
/-- A proved equation keeps the large constant division out of eager record
projection reduction in the kernel checker. -/
theorem chunkStart_exists (block : Nat) : ∃ start : Nat, start = block / gridLimit * gridLimit * 256 := ⟨_, rfl⟩
noncomputable def chunkStart (block : Nat) : Nat := Classical.choose (chunkStart_exists block)
theorem chunkStart_eq (block : Nat) : chunkStart block = block / gridLimit * gridLimit * 256 :=
  Classical.choose_spec (chunkStart_exists block)
theorem localPid_exists (block : Nat) : ∃ pid : Nat, pid = block % gridLimit := ⟨_, rfl⟩
noncomputable def localPid (block : Nat) : Nat := Classical.choose (localPid_exists block)
theorem localPid_eq (block : Nat) : localPid block = block % gridLimit :=
  Classical.choose_spec (localPid_exists block)

theorem chunk_identity (k : Nat) : localPid k * 256 + chunkStart k = k * 256 := by
  rw [localPid_eq, chunkStart_eq]
  rw [← Nat.add_mul, Nat.mul_comm (k / gridLimit) gridLimit, Nat.mod_add_div]

theorem localPid_bound (k : Nat) : localPid k < 2147483647 := by
  rw [localPid_eq]
  exact Nat.mod_lt _ (by decide)

theorem block_coverage (count o : Nat) (ho : o < count) :
    o / 256 < blocks count ∧ o % 256 < 256 ∧ o / 256 * 256 + o % 256 = o := by
  unfold blocks
  omega

theorem block_base_bound (count k : Nat) (hk : k < blocks count) : k * 256 < count := by
  unfold blocks at hk
  omega

/-- Includes inactive tail lanes. No artificial count ≤ max−255 condition. -/
theorem index_u64_bound (count k lane : Nat) (hc : count ≤ i64Max)
    (hk : k < blocks count) (hl : lane < 256) :
    k * 256 + lane < u64Mod := by
  have := block_base_bound count k hk
  unfold i64Max u64Mod at *
  omega

theorem safe_bounds (count idx : Nat) (hc : 0 < count) :
    (if idx < count then idx else 0) < count := by split <;> omega

/-- Even masked-off lanes form a valid metadata pointer: their safe index is 0.
Both metadata byte extent and source/destination extents are explicit. -/
theorem gather_bounds (n width capacity idx : Nat) (rows : Nat → Nat)
    (hw : 0 < width) (hc : 0 < n * width) (hi : idx < n * width)
    (hr : ∀ i, i < n → rows i < capacity) :
    idx / width < n ∧ idx % width < width ∧
    rows (idx / width) * width + idx % width < capacity * width := by
  have hd : idx / width < n := (Nat.div_lt_iff_lt_mul hw).mpr hi
  have hm : idx % width < width := Nat.mod_lt _ hw
  refine ⟨hd, hm, ?_⟩
  calc
    rows (idx / width) * width + idx % width < (rows (idx / width) + 1) * width := by simpa [Nat.add_mul] using Nat.add_lt_add_left hm (rows (idx / width) * width)
    _ ≤ capacity * width := Nat.mul_le_mul_right _ (hr _ hd)

theorem metadata_byte_bound (n idx byte : Nat) (hi : idx < n) (hb : byte < 8) :
    idx * 8 + byte < n * 8 := by omega

theorem address_i64 (address extent : Nat) (h : address < extent)
    (he : extent ≤ i64Max) : address < 9223372036854775808 := by
  unfold i64Max at he
  omega

theorem stored_word_coordinate (row col byte width itemBytes : Nat)
    (hc : col < width) (hb : byte < itemBytes) (he : 0 < itemBytes) :
    (row * width + col) * itemBytes + byte =
      row * (width * itemBytes) + (col * itemBytes + byte) ∧
    col * itemBytes + byte < width * itemBytes := by
  constructor
  · ring
  · calc
      col * itemBytes + byte < (col + 1) * itemBytes := by simpa [Nat.add_mul] using Nat.add_lt_add_left hb (col * itemBytes)
      _ ≤ width * itemBytes := Nat.mul_le_mul_right _ hc

#axiomsClean index_u64_bound
#axiomsClean gather_bounds
#axiomsClean stored_word_coordinate
end VeriTile.Bench.Repairs.CacheTransform
