import bench.tritonbench_g.fused_layernorm_triton.Proofs.Mathematical_WholeWelford

namespace VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
open VeriTile.Triton

/-- All columns, including the final partial chunk, lie below this bound. -/
theorem block_count_covers (D B : Nat) (hB : 0 < B) :
    D ≤ blockCount D B * B := by
  have hd := Nat.div_add_mod (D + B - 1) B
  have hm := Nat.mod_lt (D + B - 1) hB
  rw [Nat.mul_comm B _] at hd
  unfold blockCount
  have hp : 1 ≤ B := hB
  omega

theorem block_count_positive (D B : Nat) (hD : 0 < D) (hB : 0 < B) :
    0 < blockCount D B := by
  have hc := block_count_covers D B hB
  by_contra h
  have hz : blockCount D B = 0 := by omega
  simp [hz] at hc
  omega

theorem column_chunk (D B j : Nat) (hB : 0 < B) (hj : j < D) :
    j / B < blockCount D B ∧ j % B < B ∧ j / B * B + j % B = j := by
  refine ⟨?_, Nat.mod_lt j hB, ?_⟩
  · exact (Nat.div_lt_iff_lt_mul hB).mpr (lt_of_lt_of_le hj (block_count_covers D B hB))
  · simpa [Nat.mul_comm] using Nat.div_add_mod j B

theorem chunk_unique (B k l i j : Nat) (hi : i < B) (hj : j < B)
    (h : k * B + i = l * B + j) : k = l ∧ i = j := by
  have hb : 0 < B := lt_of_le_of_lt (Nat.zero_le i) hi
  have hd1 : (k * B + i) / B = k := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hb, Nat.div_eq_of_lt hi]
    simp
  have hd2 : (l * B + j) / B = l := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hb, Nat.div_eq_of_lt hj]
    simp
  have hk : k = l := by rw [← hd1, ← hd2, h]
  exact ⟨hk, by subst l; omega⟩

theorem row_column_injective (D r q i j : Nat) (hi : i < D) (hj : j < D)
    (h : i + D * r = j + D * q) : r = q ∧ i = j := by
  have h' : r * D + i = q * D + j := by simpa [Nat.mul_comm, Nat.add_comm] using h
  exact chunk_unique D r q i j hi hj h'

theorem output_bounds (S D r j : Nat) (hr : r < S) (hj : j < D) :
    j + D * r < S * D := by
  have h : D * (r + 1) ≤ D * S := Nat.mul_le_mul_left D hr
  nlinarith

/-- Both statistics metadata views index precisely the same backing cell. -/
theorem statistics_reinterpret (S r : Nat) (column : Fin 1) :
    r * 1 + column.val * S = r * 1 + column.val * 1 := by
  have hc : column.val = 0 := Nat.eq_zero_of_le_zero (by omega)
  simp [hc]

#axiomsClean column_chunk
#axiomsClean chunk_unique
#axiomsClean row_column_injective
#axiomsClean output_bounds
#axiomsClean statistics_reinterpret
end VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
