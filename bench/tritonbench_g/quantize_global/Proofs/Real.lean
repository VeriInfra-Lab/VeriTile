import bench.tritonbench_g.quantize_global.Proofs.Maximum
import bench.tritonbench_g.quantize_global.Proofs.Mathematical_RealScalar
namespace QuantizeGlobalRepair.Real
open VeriTile.Triton FP.Structural
noncomputable section
open Classical
set_option maxHeartbeats 2000000
set_option maxRecDepth 24000

/-- Independent global absolute maximum, including the empty-set value zero. -/
def Maximum (n : Nat) (xs : Nat → ℝ) (m : ℝ) : Prop :=
  0 ≤ m ∧ (∀ i, i < n → |xs i| ≤ m) ∧
    ∀ b, 0 ≤ b → (∀ i, i < n → |xs i| ≤ b) → m ≤ b

/-- Independent mathematical quantization with ties to the even integer. -/
def quantized (x m : ℝ) : Int :=
  if m = 0 then 0 else QuantizationReal.nearestEven (127 * x / m)

def arithmetic : Scalar ℝ :=
  ⟨id,Neg.neg,HMul.hMul,HDiv.hDiv,fun x y => decide (x < y),
    fun x y => decide (x = y),fun x => BitVec.ofInt 64 (QuantizationReal.nearestEven x)⟩

theorem tree_nonnegative (t : MaxTree) (xs : Nat → ℝ) : 0 ≤ t.eval 0 abs max xs := by
  induction t with
  | empty => exact le_rfl
  | leaf i => exact abs_nonneg _
  | node l r ihl ihr => exact le_trans ihl (le_max_left _ _)

theorem tree_dominates (t : MaxTree) (xs : Nat → ℝ) (i : Nat) (hi : i ∈ t.leaves) :
    |xs i| ≤ t.eval 0 abs max xs := by
  induction t with
  | empty => simp [MaxTree.leaves] at hi
  | leaf j =>
    have he : i = j := by simpa [MaxTree.leaves] using hi
    subst i
    exact le_rfl
  | node l r ihl ihr =>
    rcases List.mem_append.mp hi with hl | hr
    · exact le_trans (ihl hl) (le_max_left _ _)
    · exact le_trans (ihr hr) (le_max_right _ _)

theorem tree_le (t : MaxTree) (xs : Nat → ℝ) (b : ℝ) (hb : 0 ≤ b)
    (hle : ∀ i ∈ t.leaves, |xs i| ≤ b) : t.eval 0 abs max xs ≤ b := by
  induction t with
  | empty => exact hb
  | leaf i => exact hle i (by simp [MaxTree.leaves])
  | node l r ihl ihr =>
    exact max_le (ihl (fun i hi => hle i (List.mem_append_left _ hi)))
      (ihr (fun i hi => hle i (List.mem_append_right _ hi)))

theorem tree_maximum (t : MaxTree) (n : Nat) (xs : Nat → ℝ) (covers : t.Covers n) :
    Maximum n xs (t.eval 0 abs max xs) :=
  ⟨tree_nonnegative t xs,fun i hi => tree_dominates t xs i ((covers i).2 hi),
    fun b hb h => tree_le t xs b hb (fun i hi => h i ((covers i).1 hi))⟩

theorem scalar_bounds (x m : ℝ) (hm : 0 ≤ m) (hx : |x| ≤ m) :
    -127 ≤ scaled arithmetic x m ∧ scaled arithmetic x m ≤ 127 := by
  by_cases hz : m = 0
  · have hzero : x = 0 := abs_eq_zero.mp (le_antisymm (by simpa [hz] using hx) (abs_nonneg x))
    simp [scaled,denominator,arithmetic,hz,hzero]
  · have hp : 0 < m := lt_of_le_of_ne hm (Ne.symm hz)
    have bound : |x / m| ≤ 1 := by rw [abs_div,abs_of_pos hp]; exact (div_le_one hp).2 hx
    have lr := abs_le.mp bound
    simp [scaled,denominator,arithmetic,hz]
    constructor <;> nlinarith

theorem clamp_identity (x m : ℝ) (hm : 0 ≤ m) (hx : |x| ≤ m) :
    bounded arithmetic (scaled arithmetic x m) = scaled arithmetic x m := by
  obtain ⟨lo,hi⟩ := scalar_bounds x m hm hx
  generalize scaled arithmetic x m = value at *
  simp [bounded,arithmetic,not_lt.mpr lo,not_lt.mpr hi]

theorem nearest_bounds (v : ℝ) (lo : -127 ≤ v) (hi : v ≤ 127) :
    -127 ≤ QuantizationReal.nearestEven v ∧ QuantizationReal.nearestEven v ≤ 127 := by
  have h := abs_le.mp (QuantizationReal.nearestEven_distance v)
  have hlow : (-128 : ℝ) < (QuantizationReal.nearestEven v : ℝ) := by linarith
  have hhigh : (QuantizationReal.nearestEven v : ℝ) < 128 := by linarith
  have l : (-128 : Int) < QuantizationReal.nearestEven v := by exact_mod_cast hlow
  have u : QuantizationReal.nearestEven v < (128 : Int) := by exact_mod_cast hhigh
  omega

theorem quantized_bounds (x m : ℝ) (hm : 0 ≤ m) (hx : |x| ≤ m) :
    -127 ≤ quantized x m ∧ quantized x m ≤ 127 := by
  by_cases hz : m = 0
  · simp [quantized,hz]
  · have h := scalar_bounds x m hm hx
    have he : scaled arithmetic x m = 127*x/m := by simp [scaled,denominator,arithmetic,hz,mul_div_assoc]
    rw [he] at h
    simpa [quantized,hz] using nearest_bounds _ h.1 h.2

theorem scalar_correct (x m : ℝ) (hm : 0 ≤ m) (hx : |x| ≤ m) :
    scalar arithmetic x m = BitVec.ofInt 8 (quantized x m) := by
  have hb := scalar_bounds x m hm hx
  have hz := nearest_bounds _ hb.1 hb.2
  have cast64 : (BitVec.ofInt 64 (QuantizationReal.nearestEven (scaled arithmetic x m))).toInt =
      QuantizationReal.nearestEven (scaled arithmetic x m) := by
    apply BitVec.toInt_ofInt_eq_self (by decide) <;> norm_num <;> omega
  unfold scalar
  rw [clamp_identity x m hm hx]
  change BitVec.ofInt 8 (BitVec.ofInt 64 (QuantizationReal.nearestEven (scaled arithmetic x m))).toInt = _
  rw [cast64]
  by_cases hm0 : m = 0
  · have hx0 : x = 0 := abs_eq_zero.mp (le_antisymm (by simpa [hm0] using hx) (abs_nonneg x))
    norm_num [quantized,scaled,denominator,arithmetic,hm0,hx0,QuantizationReal.nearestEven]
  · simp [quantized,scaled,denominator,arithmetic,hm0,mul_div_assoc]

theorem signed_readback (x m : ℝ) (hm : 0 ≤ m) (hx : |x| ≤ m) :
    (scalar arithmetic x m).toInt = quantized x m := by
  rw [scalar_correct x m hm hx]
  have h := quantized_bounds x m hm hx
  apply BitVec.toInt_ofInt_eq_self (by decide) <;> norm_num <;> omega

#specNonCircular Maximum avoiding [source,kernel,scalar]
#specNonCircular quantized avoiding [source,kernel,scalar,scaled,bounded]
#axiomsClean tree_maximum
#axiomsClean scalar_correct
#axiomsClean signed_readback
end
end QuantizeGlobalRepair.Real
