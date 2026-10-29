import bench.tritonbench_g.matmul_persistent_triton.Proofs.HintCached
import bench.tritonbench_g.matmul_persistent_triton.Proofs.ScheduleCoverage

/-! Symbolic coverage of the hint sites. For positive K, every valid output
tile reaches its ki=0 hint branch. Any nonempty M/N remainder therefore yields
an actual reached counterexample; the positive witnesses are not the only
dimensions for which the property is decided. -/
namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.HintCoverage
open NativeDomain NativeBits HintMath HintReached
set_option maxHeartbeats 4000000

theorem remainder_tile_bound (bound block : Nat) (hb : 0 < block)
    (hr : bound % block ≠ 0) : bound / block < RealMath.blocks bound block := by
  have hc := RealMath.blocks_cover bound block hb
  have he := Nat.mod_add_div bound block
  have hpos : 0 < bound % block := by omega
  by_contra hh
  have hm := Nat.mul_le_mul_right block
    (show RealMath.blocks bound block ≤ bound/block by omega)
  nlinarith

theorem coordinate_reached (p : Parameters) (h : Admissible p) (hk : 0 < count p)
    (r c : Nat) (hr : r < rows p) (hc : c < columns p) :
    ∃ pid i, pid < grid p ∧ i < trip p pid ∧ i % count p = 0 ∧
      rowIndex p pid i = r ∧ columnIndex p pid i = c := by
  obtain ⟨t,ht,hrow,hcol⟩ := Schedule.grouped_covered (rows p) (columns p) p.GM r c h.gm hr hc
  obtain ⟨hp,hq,he⟩ := Schedule.tile_covered (total p) p.SMS t h.sms ht
  refine ⟨t%p.SMS,(t/p.SMS)*count p,hp,?_,?_,?_,?_⟩
  · have hh := Nat.mul_lt_mul_of_pos_right hq hk
    simpa only [trip,work,Nat.mul_comm] using hh
  · simp
  · simpa only [rowIndex,tile,Nat.mul_div_cancel _ hk,he] using hrow
  · simpa only [columnIndex,tile,Nat.mul_div_cancel _ hk,he] using hcol

theorem restart_at (p : Parameters) (h : Admissible p) (pid : Nat)
    (hp : pid < grid p) (hk : 0 < count p) (i : Nat) (hi : i < trip p pid)
    (hr : i % count p = 0) :
    ((NativeLoop.run p pid (i+1)).ki == bits 0) = Bool.true := by
  have hc := (NativeCoordinates.control_after p h pid hp hk i hi).2
  have he := eq_exact hc (bits_exact (z := 0) (by norm_num [Fits]))
  simpa [hr] using he

/-- Every genuinely partial M tail gives a legal PID and a real hint-refresh
iteration whose cached row vector violates full-BM contiguity. -/
theorem any_row_tail_counterexample (p : Parameters) (h : Admissible p)
    (hk : 0 < count p) (ht : p.M % p.BM ≠ 0) :
    ∃ pid i, pid < grid p ∧ i < trip p pid ∧
      ((NativeLoop.run p pid (i+1)).ki == bits 0) = Bool.true ∧
      ¬ Contiguous p.BM (HintCached.rowValues p pid i) := by
  have hr := remainder_tile_bound p.M p.BM h.bm ht
  obtain ⟨pid,i,hp,hi,hphase,hrow,_hcol⟩ :=
    coordinate_reached p h hk (p.M/p.BM) 0 hr (columns_pos p h)
  refine ⟨pid,i,hp,hi,restart_at p h pid hp hk i hi hphase,?_⟩
  rw [(HintCached.values_match p h pid hp hk i hi).1]
  apply partial_row_not_contiguous p h pid hp hk i hi (p.M%p.BM)
  · rw [hrow]
    simpa [Nat.add_comm,Nat.mul_comm] using (Nat.mod_add_div p.M p.BM).symm
  · omega
  · exact Nat.mod_lt _ h.bm

theorem any_column_tail_counterexample (p : Parameters) (h : Admissible p)
    (hk : 0 < count p) (ht : p.N % p.BN ≠ 0) :
    ∃ pid i, pid < grid p ∧ i < trip p pid ∧
      ((NativeLoop.run p pid (i+1)).ki == bits 0) = Bool.true ∧
      ¬ Contiguous p.BN (HintCached.columnValues p pid i) := by
  have hc := remainder_tile_bound p.N p.BN h.bn ht
  obtain ⟨pid,i,hp,hi,hphase,_hrow,hcol⟩ :=
    coordinate_reached p h hk 0 (p.N/p.BN) (rows_pos p h) hc
  refine ⟨pid,i,hp,hi,restart_at p h pid hp hk i hi hphase,?_⟩
  rw [(HintCached.values_match p h pid hp hk i hi).2]
  apply partial_column_not_contiguous p h pid hp hk i hi (p.N%p.BN)
  · rw [hcol]
    simpa [Nat.add_comm,Nat.mul_comm] using (Nat.mod_add_div p.N p.BN).symm
  · omega
  · exact Nat.mod_lt _ h.bn

/-- Exact criterion for these two one-dimensional coordinate promises over
all reached native-projection iterations when K is positive. This statement
does not narrow the public mathematical theorem or certify backend uses. -/
theorem all_reached_promises_iff (p : Parameters) (h : Admissible p) (hk : 0 < count p) :
    (∀ pid, pid < grid p → ∀ i, i < trip p pid →
      TilePromises p.BM (HintCached.rowValues p pid i) ∧
      TilePromises p.BN (HintCached.columnValues p pid i)) ↔
    p.BM ∣ p.M ∧ p.BN ∣ p.N := by
  constructor
  · intro ha
    constructor
    · apply Nat.dvd_of_mod_eq_zero
      by_contra ht
      obtain ⟨pid,i,hp,hi,_hs,hbad⟩ := any_row_tail_counterexample p h hk ht
      exact hbad (ha pid hp i hi).1.2.2
    · apply Nat.dvd_of_mod_eq_zero
      by_contra ht
      obtain ⟨pid,i,hp,hi,_hs,hbad⟩ := any_column_tail_counterexample p h hk ht
      exact hbad (ha pid hp i hi).2.2.2
  · rintro ⟨hm,hn⟩ pid hp i hi
    exact HintCached.reached_promises p h hm hn pid hp hk i hi

end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.HintCoverage
