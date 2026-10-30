import bench.tritonbench_g.batched_vecmat_mult.Proofs.Mathematical_HostGrid

namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.ValueProofs
open VeriTile.Triton

/-- Every unmasked tile lane launched by the floor grid is in bounds. -/
theorem tiled_coordinate_lt (size block : Nat) (pid : Fin (size / block)) (lane : Fin block) :
    pid.val * block + lane.val < size := by
  have hb := Nat.mul_le_mul_right block (Nat.succ_le_of_lt pid.isLt)
  rw [Nat.succ_mul] at hb
  have hf := Nat.div_mul_le_self size block
  have hl := lane.isLt
  omega

theorem row_major_bound {rows cols i j : Nat} (hi : i < rows) (hj : j < cols) :
    i * cols + j < rows * cols := by
  have hb := Nat.mul_le_mul_right cols (Nat.succ_le_of_lt hi)
  rw [Nat.succ_mul] at hb
  omega

theorem all_unmasked_addresses_in_declared_extents (p : Parameters)
    (m : Fin (gridM p)) (n : Fin (gridN p)) (c : Fin (p.K / p.BK))
    (i : Fin p.BM) (j : Fin p.BN) (k : Fin p.BK) :
    (m.val*p.BM+i.val)*p.K + (c.val*p.BK+k.val) < p.M*p.K ∧
    (m.val*p.BM+i.val)*p.N*p.K + (n.val*p.BN+j.val)*p.K + (c.val*p.BK+k.val) < p.M*p.N*p.K ∧
    (m.val*p.BM+i.val)*p.N + (n.val*p.BN+j.val) < p.M*p.N := by
  have hm := tiled_coordinate_lt p.M p.BM m i
  have hn := tiled_coordinate_lt p.N p.BN n j
  have hk := tiled_coordinate_lt p.K p.BK c k
  have hout := row_major_bound hm hn
  refine ⟨row_major_bound hm hk, ?_, hout⟩
  simpa [Nat.add_mul] using row_major_bound hout hk

/-- Divisibility closes coverage, not just bounds: every declared output
coordinate belongs to an actual launched block and active unmasked lane. -/
theorem complete_output_coverage (p : Parameters) (hBM : 0 < p.BM) (hBN : 0 < p.BN)
    (hM : p.BM ∣ p.M) (hN : p.BN ∣ p.N) (r : Fin p.M) (s : Fin p.N) :
    ∃ m : Fin (gridM p), ∃ n : Fin (gridN p), ∃ i : Fin p.BM, ∃ j : Fin p.BN,
      m.val*p.BM+i.val = r.val ∧ n.val*p.BN+j.val = s.val := by
  have hm : r.val / p.BM < gridM p := by
    apply (Nat.div_lt_iff_lt_mul hBM).2
    simpa [gridM, Nat.div_mul_cancel hM] using r.isLt
  have hn : s.val / p.BN < gridN p := by
    apply (Nat.div_lt_iff_lt_mul hBN).2
    simpa [gridN, Nat.div_mul_cancel hN] using s.isLt
  exact ⟨⟨r.val/p.BM,hm⟩,⟨s.val/p.BN,hn⟩,
    ⟨r.val%p.BM,Nat.mod_lt _ hBM⟩,⟨s.val%p.BN,Nat.mod_lt _ hBN⟩,
    by simpa [Nat.mul_comm] using Nat.div_add_mod r.val p.BM,
    by simpa [Nat.mul_comm] using Nat.div_add_mod s.val p.BN⟩

theorem complete_inner_coverage (p : Parameters) (hBK : 0 < p.BK)
    (hK : p.BK ∣ p.K) (k : Fin p.K) :
    ∃ c : Fin (p.K / p.BK), ∃ lane : Fin p.BK, c.val*p.BK+lane.val = k.val := by
  have hc : k.val / p.BK < p.K / p.BK := by
    apply (Nat.div_lt_iff_lt_mul hBK).2
    simpa [Nat.div_mul_cancel hK] using k.isLt
  exact ⟨⟨k.val/p.BK,hc⟩,⟨k.val%p.BK,Nat.mod_lt _ hBK⟩,
    by simpa [Nat.mul_comm] using Nat.div_add_mod k.val p.BK⟩

#axiomsClean tiled_coordinate_lt
#axiomsClean row_major_bound
#axiomsClean all_unmasked_addresses_in_declared_extents
#axiomsClean complete_output_coverage
#axiomsClean complete_inner_coverage
end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.ValueProofs
