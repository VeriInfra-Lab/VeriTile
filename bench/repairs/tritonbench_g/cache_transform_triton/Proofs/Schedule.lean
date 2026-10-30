import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.Bounds

namespace VeriTile.Bench.Repairs.CacheTransform
set_option maxRecDepth 10000

/-- Every mathematical block belongs to a real host chunk, including the
short last chunk; its local PID is below the exact Python min-grid extent. -/
theorem host_chunk_membership (count k : Nat) (hk : k < blocks count) :
    let first := k / gridLimit * gridLimit
    first < blocks count ∧ localPid k < min (blocks count - first) gridLimit ∧
    first + localPid k = k ∧ chunkStart k = first * 256 := by
  dsimp
  rw [localPid_eq, chunkStart_eq]
  have hdiv := Nat.mod_add_div k gridLimit
  have hmod := Nat.mod_lt k (show 0 < gridLimit by decide)
  rw [Nat.mul_comm gridLimit (k / gridLimit)] at hdiv
  constructor
  · omega
  constructor
  · exact Nat.lt_min.mpr ⟨by omega, hmod⟩
  constructor
  · omega
  · rfl

/-- Exact ownership, as used by the masked byte store. -/
theorem byte_owner_unique (o k j a b : Nat) (ha : a < 256) (hb : b < 256)
    (h₁ : k * 256 + a = o) (h₂ : j * 256 + b = o) : k = j ∧ a = b := by omega

#axiomsClean host_chunk_membership
#axiomsClean byte_owner_unique
end VeriTile.Bench.Repairs.CacheTransform
