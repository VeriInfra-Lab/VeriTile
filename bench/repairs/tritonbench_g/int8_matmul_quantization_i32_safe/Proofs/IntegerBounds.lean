import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Contract
import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.KernelNativeModel
import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.KernelIntegerIdeal

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe
noncomputable section
set_option maxHeartbeats 4000000

def Fits32 (x : Int) : Prop := -(2^31) ≤ x ∧ x < 2^31

theorem byte_bounds (x : Byte) : -128 ≤ x.toInt ∧ x.toInt ≤ 127 := by
  have lo := BitVec.le_toInt (x := x)
  have hi := BitVec.toInt_le (x := x)
  norm_num at lo hi
  exact ⟨lo, hi⟩

theorem product_bounds (x y : Byte) : -16384 ≤ x.toInt*y.toInt ∧ x.toInt*y.toInt ≤ 16384 := by
  obtain ⟨hx, hx'⟩ := byte_bounds x
  obtain ⟨hy, hy'⟩ := byte_bounds y
  constructor
  · nlinarith [mul_nonneg (show 0 ≤ x.toInt+128 by omega) (show 0 ≤ y.toInt+128 by omega),
      mul_nonneg (show 0 ≤ 128-x.toInt by omega) (show 0 ≤ 128-y.toInt by omega)]
  · nlinarith [mul_nonneg (show 0 ≤ x.toInt+128 by omega) (show 0 ≤ 128-y.toInt by omega),
      mul_nonneg (show 0 ≤ 128-x.toInt by omega) (show 0 ≤ y.toInt+128 by omega)]

theorem sum_bounds (xs : List Int) (h : ∀ x ∈ xs, -16384 ≤ x ∧ x ≤ 16384) :
    -(16384 * (xs.length : Int)) ≤ xs.sum ∧ xs.sum ≤ 16384 * (xs.length : Int) := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
    have hx := h x (by simp)
    have ht := ih (fun y hy => h y (by simp [hy]))
    simp only [List.sum_cons, List.length_cons, Nat.cast_add, Nat.cast_one]
    constructor <;> linarith

theorem sum_fits (xs : List Int) (hk : xs.length ≤ 131071)
    (h : ∀ x ∈ xs, -16384 ≤ x ∧ x ≤ 16384) : Fits32 xs.sum := by
  obtain ⟨lo, hi⟩ := sum_bounds xs h
  have hn : (xs.length : Int) ≤ 131071 := by exact_mod_cast hk
  norm_num [Fits32]
  constructor <;> omega

def maskedProduct (K : Nat) (a b : Nat → Byte) (i : Nat) : Int :=
  if i < K then (a i).toInt*(b i).toInt else 0

theorem tail_zero (K : Nat) (a b : Nat → Byte) (i : Nat) (h : K ≤ i) :
    maskedProduct K a b i = 0 := by simp [maskedProduct, show ¬ i < K by omega]

theorem masked_sum_filter (K : Nat) (a b : Nat → Byte) (indices : List Nat) :
    (indices.map (maskedProduct K a b)).sum =
      ((indices.filter (fun i => i < K)).map (fun i => (a i).toInt*(b i).toInt)).sum := by
  induction indices with
  | nil => rfl
  | cons i is ih =>
    by_cases h : i < K <;> simp [maskedProduct, h, ih]

/-- Any order, any partial subset, any number of zero padding lanes. Only
the distinct active source products count toward the K bound. -/
theorem arbitrary_prefix_fits (K : Nat) (hk : K ≤ 131071) (a b : Nat → Byte)
    (indices : List Nat) (distinct : indices.Nodup) :
    Fits32 (indices.map (maskedProduct K a b)).sum := by
  rw [masked_sum_filter]
  apply sum_fits
  · rw [List.length_map]
    have subset : indices.filter (fun i => i < K) ⊆ List.range K := by
      intro i hi
      exact List.mem_range.mpr (by simpa using (List.mem_filter.mp hi).2)
    have count := ((distinct.filter _).subperm subset).length_le
    simpa using count.trans (by simpa using hk)
  · intro x hx
    obtain ⟨i, _, rfl⟩ := List.mem_map.mp hx
    exact product_bounds _ _

theorem clamp_exact (x : Int) (h : Fits32 x) : KernelNativeModel.clampSigned 32 x = x := by
  unfold KernelNativeModel.clampSigned
  norm_num
  have bounds := h
  norm_num [Fits32] at bounds
  rw [min_eq_right (by omega), max_eq_right (by omega)]

theorem encode_exact (x : Int) (h : Fits32 x) : (BitVec.ofInt 32 x).toInt = x := by
  exact BitVec.toInt_ofInt_eq_self (by decide) h.1 h.2

/-- The documented saturating and wrapping K32 families coincide with the
exact integer total whenever the proved prefix bound applies. -/
theorem instruction_exact (kind : ISA) (xs ys : Fin 32 → Byte) (old : BitVec 32)
    (h : Fits32 (KernelNativeModel.instructionTotal xs ys old)) :
    (KernelNativeModel.instructions.result kind xs ys old).toInt =
      KernelNativeModel.instructionTotal xs ys old := by
  cases kind <;> simp only [KernelNativeModel.instructions]
  · rw [clamp_exact _ h, encode_exact _ h]
  · exact encode_exact _ h

theorem source_index_injective (c : Config) (z : Fin c.S) :
    Function.Injective (fun q : Nat × Fin c.BK => sourceK c q.1 z q.2) := by
  intro x y h
  obtain ⟨ht, _, hl⟩ := Int8DequantMatmul.sourceK_unique c.toConfig x.1 z x.2 y.1 z y.2
    z.isLt z.isLt x.2.isLt y.2.isLt h
  exact Prod.ext ht (Fin.ext hl)

theorem source_prefix_fits (s : Shape) (hk : s.K ≤ 131071) (c : Config) (z : Fin c.S)
    (a b : Nat → Byte) (coordinates : List (Nat × Fin c.BK)) (distinct : coordinates.Nodup) :
    Fits32 ((coordinates.map (fun q => sourceK c q.1 z q.2)).map (maskedProduct s.K a b)).sum := by
  apply arbitrary_prefix_fits _ hk
  exact distinct.map (source_index_injective c z)

/-- The exact operand stream of the selected finite micro-instruction plans.
Missing K32 lanes are explicit zeros and are absent from Micro.terms. -/
def planCoordinates (BK steps : Nat) (plans : Nat → DotPlan BK) : List (Nat × Fin BK) :=
  (List.range steps).flatMap (fun t =>
    ((plans t).operations.flatMap Int8DequantMatmul.Micro.terms).map (fun lane => (t,lane)))

theorem planCoordinates_distinct (BK steps : Nat) (plans : Nat → DotPlan BK) :
    (planCoordinates BK steps plans).Nodup := by
  unfold planCoordinates
  apply List.nodup_flatMap.mpr
  constructor
  · intro t _
    have unique := (plans t).complete.nodup_iff.mpr (List.nodup_finRange BK)
    exact unique.map (fun _ _ h => (Prod.mk.inj h).2)
  · exact (List.nodup_range (n := steps)).imp (fun {t u} ne x hx hy => by
      obtain ⟨l, _, hl⟩ := List.mem_map.mp hx
      obtain ⟨m, _, hm⟩ := List.mem_map.mp hy
      exact ne (congrArg Prod.fst (hl.trans hm.symm)))

/-- Every micro-instruction boundary, scalar prefix, entire block, and K-loop
prefix is covered by a prefix of this stream, independently of values. -/
theorem every_plan_prefix_fits (s : Shape) (hk : s.K ≤ 131071) (c : Config) (z : Fin c.S)
    (a b : Nat → Byte) (plans : Nat → DotPlan c.BK) (iterations cut : Nat) :
    Fits32 ((((planCoordinates c.BK iterations plans).take cut).map
      (fun q => sourceK c q.1 z q.2)).map (maskedProduct s.K a b)).sum := by
  apply source_prefix_fits _ hk
  exact (planCoordinates_distinct c.BK iterations plans).take

theorem source_mask_exact (K : Nat) (c : Config) (t z l : Nat) :
    (z*c.BK+l < K-t*(c.BK*c.S)) ↔ sourceK c t z l < K := by
  simp only [sourceK, Int8DequantMatmul.sourceK]
  omega

theorem next_dimension_overflows : ¬ Fits32 ((131072 : Int)*((-128)*(-128))) := by
  norm_num [Fits32]

#axiomsClean product_bounds
#axiomsClean arbitrary_prefix_fits
#axiomsClean tail_zero
#axiomsClean instruction_exact
#axiomsClean source_prefix_fits
#axiomsClean planCoordinates_distinct
#axiomsClean every_plan_prefix_fits
#axiomsClean source_mask_exact
#axiomsClean next_dimension_overflows
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe
