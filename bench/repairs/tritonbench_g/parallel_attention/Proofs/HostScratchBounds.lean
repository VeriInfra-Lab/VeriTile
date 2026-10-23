import bench.repairs.tritonbench_g.parallel_attention.Proofs.MachineClosure

namespace VeriTile.Bench.Repairs.ParallelAttention.Machine
open VeriTile.Bench.Repairs.ParallelAttention

/-- FP32 byte size of either contiguous NV-sharded DQ or DK scratch tensor. -/
def qkScratchBytes (s : Shape) := 4*(valueShards s*s.B*s.H*s.T*s.K)

/-- For V at least 512, the formed-index guard also bounds the complete
DQ/DK scratch byte span, because 4*K is at most V. No nonempty premise is needed. -/
theorem large_value_qk_bytes (s : Shape) (h : ShapeContract s) (g : Guard s)
    (hv : 512 ≤ s.V) : 4*(valueShards s*s.B*s.H*s.T*s.K) < 2^31 := by
  have hk := h.key_limit
  have hw : 4*s.K ≤ largestWidth s :=
    le_trans (by omega : 4*s.K ≤ s.V) (le_max_right s.K s.V)
  have hm := Nat.mul_le_mul_left (valueShards s*s.B*s.H*s.T) hw
  have hg := g.formed
  dsimp only [formedLimit] at hg
  nlinarith

/-- The source-selected BV makes every reduction below V=512 at most four-way. -/
theorem small_value_shards (s : Shape) (h : ShapeContract s) (hv : s.V < 512) :
    valueShards s ≤ 4 := by
  unfold valueShards
  rw [h.value_tile]
  unfold selectedTile
  split_ifs <;> unfold blocks <;> omega

/-- A scratch byte span reaching the signed 32-bit limit can only have at
most four reduction inputs. This is arithmetic, not a Torch dispatch theorem. -/
theorem large_qk_bytes_small_shards (s : Shape) (h : ShapeContract s) (g : Guard s)
    (hb : 2^31 ≤ qkScratchBytes s) : valueShards s ≤ 4 := by
  apply small_value_shards s h
  by_contra hv
  have bounded := large_value_qk_bytes s h g (by omega)
  change 2^31 ≤ 4*(valueShards s*s.B*s.H*s.T*s.K) at hb
  omega

theorem qk_bytes_or_small_shards (s : Shape) (h : ShapeContract s) (g : Guard s) :
    qkScratchBytes s < 2^31 ∨ valueShards s ≤ 4 := by
  by_cases hb : qkScratchBytes s < 2^31
  · exact Or.inl hb
  · exact Or.inr (large_qk_bytes_small_shards s h g (by omega))

/-- Connect the byte-span statement to the existing scratch output tags. -/
theorem query_key_bytes (s : Shape) (kind : OutputKind) (hk : kind = .query ∨ kind = .key) :
    4*kind.count s = qkScratchBytes s := by
  rcases hk with rfl|rfl <;>
    simp only [OutputKind.count,OutputKind.shards,OutputKind.width,qkScratchBytes,valueShards] <;>
    ring

theorem query_key_split_small (s : Shape) (h : ShapeContract s) (g : Guard s)
    (kind : OutputKind) (hk : kind = .query ∨ kind = .key)
    (hb : 2^31 ≤ 4*kind.count s) : kind.shards s ≤ 4 := by
  rw [query_key_bytes s kind hk] at hb
  have bound := large_qk_bytes_small_shards s h g hb
  rcases hk with rfl|rfl <;> exact bound

#axiomsClean large_value_qk_bytes
#axiomsClean small_value_shards
#axiomsClean large_qk_bytes_small_shards
#axiomsClean qk_bytes_or_small_shards
#axiomsClean query_key_bytes
#axiomsClean query_key_split_small
#specNonCircular qkScratchBytes avoiding [combinedRun,hostRun,wordReference]
end VeriTile.Bench.Repairs.ParallelAttention.Machine
