import bench.repairs.tritonbench_g.chunk_gla_simple_min_dot_tile.Proofs.Address

/-! Native i64 address and mask arithmetic. Storage limits below are the
ordinary FP32 allocation byte-size limit. `arangeFits` records the compiler's
existing arange setup check (end >> 32 == 0), not an added wrapper restriction.
Inactive pointer expressions may wrap; the proof covers every mask comparison
and every active load/store, exactly where the source uses the result. -/
namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.Native
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaSimple Math Real
open VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.Host
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

abbrev word (n : Nat) : BitVec 64 := BitVec.ofNat 64 n

def Fits (c : Shape) : Prop := ∀ r, 4 * capacity c r < 2^63

theorem element_bound (c : Shape) (hf : Fits c) (r : RegionName) : capacity c r < 2^61 := by
  have h := hf r
  norm_num at h ⊢
  omega

theorem word_exact (n : Nat) (h : n < 2^63) : (word n).toInt = (n : Int) := by
  have hn : n < 2^64 := lt_trans h (by decide)
  have hnat : (word n).toNat = n := Nat.mod_eq_of_lt hn
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [hnat]; omega),hnat]

theorem comparison_exact (a b : Nat) (ha : a < 2^63) (hb : b < 2^63) :
    (word a).toInt < (word b).toInt ↔ a < b := by
  rw [word_exact a ha,word_exact b hb]
  exact Int.ofNat_lt

theorem dimensions_fit (c : Shape) (hf : Fits c) (hn : 0 < c.heads*c.T*c.V) :
    c.T < 2^61 ∧ c.K < 2^61 ∧ c.V < 2^61 := by
  have hh : 0 < c.heads := by
    by_contra h
    have he : c.heads=0 := by omega
    simp [he] at hn
  have ht : 0 < c.T := by
    by_contra h
    have he : c.T=0 := by omega
    simp [he] at hn
  have hht : 0 < c.heads*c.T := Nat.mul_pos hh ht
  have hg : c.heads*c.T < 2^61 := by simpa [capacity] using element_bound c hf "g"
  have hq : c.heads*c.T*c.K < 2^61 := by simpa [capacity] using element_bound c hf "q"
  have hv : c.heads*c.T*c.V < 2^61 := by simpa [capacity] using element_bound c hf "v"
  exact ⟨lt_of_le_of_lt (Nat.le_mul_of_pos_left _ hh) hg,
    lt_of_le_of_lt (Nat.le_mul_of_pos_left _ hht) hq,
    lt_of_le_of_lt (Nat.le_mul_of_pos_left _ hht) hv⟩

theorem padded_coordinate (n b pid lane : Nat) (hp : pid < (n+b-1)/b) (hi : lane < b) :
    pid*b+lane < n+b := by
  have hdiv := Nat.div_mul_le_self (n+b-1) b
  have hmul := Nat.mul_le_mul_right b (Nat.succ_le_of_lt hp)
  simp only [Nat.succ_mul] at hmul
  omega

theorem coordinate_fits (n b pid lane : Nat) (hn : n < 2^61) (hb : b < 2^32)
    (hp : pid < (n+b-1)/b) (hi : lane < b) : pid*b+lane < 2^63 := by
  have h := padded_coordinate n b pid lane hp hi
  norm_num at hn hb ⊢
  omega

theorem coordinate_word (n b pid lane : Nat) (hn : n < 2^61) (hb : b < 2^32)
    (hp : pid < (n+b-1)/b) (hi : lane < b) :
    (word pid * word b + word lane).toInt = (pid*b+lane : Nat) := by
  rw [← BitVec.ofNat_mul,← BitVec.ofNat_add]
  exact word_exact _ (coordinate_fits n b pid lane hn hb hp hi)

/-- Both padded sides of every rectangular mask retain their signed meaning. -/
theorem mask_exact (n b pid lane : Nat) (hn : n < 2^61) (hb : b < 2^32)
    (hp : pid < (n+b-1)/b) (hi : lane < b) :
    (word pid * word b + word lane).toInt < (word n).toInt ↔ pid*b+lane < n := by
  rw [← BitVec.ofNat_mul,← BitVec.ofNat_add]
  exact comparison_exact _ _ (coordinate_fits n b pid lane hn hb hp hi) (lt_trans hn (by decide))

/-- Source row, column and key coordinates, including inactive tail lanes. -/
theorem all_coordinates (c : Shape) (hf : Fits c) (hn : 0 < c.heads*c.T*c.V)
    (arangeFits : c.BT < 2^32) (idx : GridIndex (grid c)) :
    (∀ i : Fin c.BT, (launchIndex c idx).chunk*c.BT+i.val < 2^63) ∧
    (∀ j : Fin c.BV, (launchIndex c idx).value*c.BV+j.val < 2^63) ∧
    (∀ kb, kb < numBlocks (layout c) → ∀ e : Fin c.BK, kb*c.BK+e.val < 2^63) := by
  obtain ⟨ht,hk,hv⟩ := dimensions_fit c hf hn
  obtain ⟨hiv,hit,_⟩ := launch_lt c idx
  have hbk : c.BK < 2^32 := lt_of_le_of_lt c.bk_max (by decide)
  have hbv : c.BV < 2^32 := lt_of_le_of_lt (Nat.min_le_right _ _) (by decide)
  refine ⟨fun i => coordinate_fits _ _ _ _ ht arangeFits hit i.isLt,
    fun j => coordinate_fits _ _ _ _ hv hbv hiv j.isLt,?_⟩
  intro kb hkb e
  exact coordinate_fits _ _ _ _ hk hbk hkb e.isLt

/-- Exact native affine offset, with a single mathematical bound on the active
sum. Since all terms are natural, every source prefix is also representable. -/
theorem affine_exact (base x stride y : Nat) (h : base+x*stride+y < 2^63) :
    (word base + word x * word stride + word y).toInt = (base+x*stride+y : Nat) := by
  simp only [← BitVec.ofNat_mul,← BitVec.ofNat_add]
  exact word_exact _ h

theorem query_exact (c : Shape) (hf : Fits c) (l : Launch) (hl : l.head < c.heads)
    (row key : Nat) (hr : row < c.T) (hk : key < c.K) :
    (word l.head * (word c.T * word c.K) + word row * word c.K + word key).toInt =
      (l.head*(c.T*c.K)+row*c.K+key : Nat) := by
  have hc : c.heads*c.T*c.K < 2^61 := by simpa [capacity] using element_bound c hf "q"
  have ha := triple_lt hl hr hk
  simp only [← BitVec.ofNat_mul,← BitVec.ofNat_add]
  exact word_exact _ (lt_trans ha (lt_trans hc (by decide)))

theorem key_exact (c : Shape) (hf : Fits c) (l : Launch) (hl : l.head < c.heads)
    (row key : Nat) (hr : row < c.T) (hk : key < c.K) :
    (word l.head * (word c.T * word c.K) + word key + word row * word c.K).toInt =
      (l.head*(c.T*c.K)+key+row*c.K : Nat) := by
  have hc : c.heads*c.T*c.K < 2^61 := by simpa [capacity] using element_bound c hf "k"
  have ha := triple_lt hl hr hk
  have hb : l.head*(c.T*c.K)+key+row*c.K < 2^63 := by
    calc _ = l.head*(c.T*c.K)+row*c.K+key := by omega
         _ < c.heads*c.T*c.K := ha
         _ < 2^63 := lt_trans hc (by decide)
  simp only [← BitVec.ofNat_mul,← BitVec.ofNat_add]
  exact word_exact _ hb

theorem history_exact (c : Shape) (hf : Fits c) (l : Launch)
    (hh : l.head < c.heads) (ht : l.chunk < c.NT)
    (key col : Nat) (hk : key < c.K) (hv : col < c.V) :
    (word l.head * (word c.NT * word c.K * word c.V) +
      word l.chunk * word c.K * word c.V + word key * word c.V + word col).toInt =
      (l.head*(c.NT*c.K*c.V)+l.chunk*c.K*c.V+key*c.V+col : Nat) := by
  have hc : c.heads*c.NT*c.K*c.V < 2^61 := by simpa [capacity] using element_bound c hf "h"
  have ha := pair_lt (pair_lt (pair_lt hh ht) hk) hv
  have hb : l.head*(c.NT*c.K*c.V)+l.chunk*c.K*c.V+key*c.V+col < 2^63 := by
    calc _ = ((l.head*c.NT+l.chunk)*c.K+key)*c.V+col := by ring
         _ < c.heads*c.NT*c.K*c.V := ha
         _ < 2^63 := lt_trans hc (by decide)
  simp only [← BitVec.ofNat_mul,← BitVec.ofNat_add]
  exact word_exact _ hb

theorem value_output_exact (c : Shape) (hf : Fits c) (l : Launch) (hl : l.head < c.heads)
    (row col : Nat) (hr : row < c.T) (hv : col < c.V) :
    (word l.head * (word c.T * word c.V) + word row * word c.V + word col).toInt =
      (l.head*(c.T*c.V)+row*c.V+col : Nat) := by
  have hc : c.heads*c.T*c.V < 2^61 := by simpa [capacity] using element_bound c hf "v"
  have ha := triple_lt hl hr hv
  simp only [← BitVec.ofNat_mul,← BitVec.ofNat_add]
  exact word_exact _ (lt_trans ha (lt_trans hc (by decide)))

theorem gate_exact (c : Shape) (hf : Fits c) (l : Launch) (hl : l.head < c.heads)
    (row : Nat) (hr : row < c.T) :
    (word l.head * word c.T + word row).toInt = (l.head*c.T+row : Nat) := by
  have hc : c.heads*c.T < 2^61 := by simpa [capacity] using element_bound c hf "g"
  have ha := pair_lt hl hr
  simp only [← BitVec.ofNat_mul,← BitVec.ofNat_add]
  exact word_exact _ (lt_trans ha (lt_trans hc (by decide)))

/-- The repaired max(K,1) host choice agrees with the retained Lean BK,
including the newly defined zero-key case. -/
theorem python_bk (c : Shape) : max 16 (min (nextPowerOfTwo (max c.K 1)) 64) = c.BK := by
  by_cases hk : c.K = 0
  · simp [hk,Shape.BK,nextPowerOfTwo]
  · have hp : 1 ≤ c.K := by omega
    simp [Nat.max_eq_left hp,Shape.BK]

/-- The host-evaluated cdiv numerator and key induction value fit i64 too. -/
theorem loop_bound (c : Shape) (hf : Fits c) (hn : 0 < c.heads*c.T*c.V) :
    c.K+c.BK-1 < 2^63 ∧ (c.K+c.BK-1)/c.BK < 2^63 := by
  have hk := (dimensions_fit c hf hn).2.1
  have hb : c.BK ≤ 64 := c.bk_max
  have h : c.K+c.BK-1 < 2^63 := by norm_num at hk ⊢; omega
  exact ⟨h,lt_of_le_of_lt (Nat.div_le_self _ _) h⟩

/-- The pointer element-to-byte scale also remains exact on every active FP32
cell; it follows from storage capacity, not a second address restriction. -/
theorem byte_offset_exact (c : Shape) (hf : Fits c) (r : RegionName) (off : Nat)
    (ho : off < capacity c r) : (word off * word 4).toInt = (off*4 : Nat) := by
  have hc := element_bound c hf r
  have h : off*4 < 2^63 := by norm_num at hc ⊢; omega
  rw [← BitVec.ofNat_mul]
  exact word_exact _ h

#print axioms all_coordinates
#print axioms history_exact
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.Native
