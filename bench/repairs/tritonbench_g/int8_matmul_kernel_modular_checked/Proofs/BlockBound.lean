import bench.tritonbench_g.int8_matmul_kernel.Whole001.Proofs.Numeric
import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.WordAdd

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
noncomputable section
set_option maxHeartbeats 4000000

theorem byte_signed_bounds (x : Byte) : -128 ≤ x.toInt ∧ x.toInt ≤ 127 := by
  have hl := @BitVec.le_two_mul_toInt 8 x
  have hu := @BitVec.toInt_lt 8 x
  norm_num only [Nat.reducePow, Nat.reduceSub, Int.reducePow] at hl hu
  omega

theorem weighted_product_bounds (x : Byte) (weight : Int)
    (hw : -1 ≤ weight ∧ weight ≤ 2) :
    -256 ≤ x.toInt * weight ∧ x.toInt * weight ≤ 254 := by
  have hx := byte_signed_bounds x
  have cases : weight = -1 ∨ weight = 0 ∨ weight = 1 ∨ weight = 2 := by omega
  rcases cases with h | h | h | h <;> rw [h] <;> constructor <;> nlinarith

theorem weighted_sum_bounds {n : Nat} (xs : Fin n → Byte) (weights : Fin n → Int)
    (hw : ∀ i, -1 ≤ weights i ∧ weights i ≤ 2) :
    -(256 * (n : Int)) ≤ ∑ i, (xs i).toInt * weights i ∧
      (∑ i, (xs i).toInt * weights i) ≤ 256 * (n : Int) := by
  constructor
  · calc
      -(256 * (n : Int)) = ∑ _i : Fin n, (-256 : Int) := by simp; ring
      _ ≤ ∑ i, (xs i).toInt * weights i :=
        Finset.sum_le_sum fun i _ => (weighted_product_bounds (xs i) (weights i) (hw i)).1
  · calc
      (∑ i, (xs i).toInt * weights i) ≤ ∑ _i : Fin n, (256 : Int) :=
        Finset.sum_le_sum fun i _ => le_trans (weighted_product_bounds (xs i) (weights i) (hw i)).2 (by decide)
      _ = 256 * (n : Int) := by simp; ring

/-- Any accumulation prefix with at most 128 admitted products is strictly
inside signed32, independent of signs or the internal order of the products. -/
theorem bounded_prefix_representable {n : Nat} (hn : n ≤ 128)
    (xs : Fin n → Byte) (weights : Fin n → Int)
    (hw : ∀ i, -1 ≤ weights i ∧ weights i ≤ 2) :
    -2147483648 ≤ ∑ i, (xs i).toInt * weights i ∧
      (∑ i, (xs i).toInt * weights i) ≤ 2147483647 := by
  have bounds := weighted_sum_bounds xs weights hw
  have hn' : (n : Int) ≤ 128 := by exact_mod_cast hn
  constructor <;> omega

theorem referenceInstruction_exact (kind : Instruction)
    (xs ys : Fin 32 → Byte) (seed : Int)
    (hseed : -2147483648 ≤ seed ∧ seed ≤ 2147483647)
    (hresult : -2147483648 ≤ seed + ∑ i, (xs i).toInt * (ys i).toInt ∧
      seed + (∑ i, (xs i).toInt * (ys i).toInt) ≤ 2147483647) :
    referenceInstruction kind xs ys (BitVec.ofInt 32 seed) =
      BitVec.ofInt 32 (seed + ∑ i, (xs i).toInt * (ys i).toInt) := by
  have hs : (BitVec.ofInt 32 seed).toInt = seed := by
    apply BitVec.toInt_ofInt_eq_self (by decide) <;> norm_num <;> omega
  unfold referenceInstruction
  rw [hs]
  dsimp only
  split_ifs
  · rw [min_eq_right hresult.2, max_eq_right hresult.1]
  · rfl

def chunkSum (xs ys : Nat → Fin 32 → Byte) (group : Nat) : Int :=
  ∑ i, (xs group i).toInt * (ys group i).toInt

def blockPrefix (xs ys : Nat → Fin 32 → Byte) (count : Nat) : Int :=
  ∑ group ∈ Finset.range count, chunkSum xs ys group

theorem blockPrefix_bounds (xs ys : Nat → Fin 32 → Byte) (count : Nat)
    (hw : ∀ group i, -1 ≤ (ys group i).toInt ∧ (ys group i).toInt ≤ 2) :
    -(8192 * (count : Int)) ≤ blockPrefix xs ys count ∧
      blockPrefix xs ys count ≤ 8192 * (count : Int) := by
  have bounds (group : Nat) : -8192 ≤ chunkSum xs ys group ∧ chunkSum xs ys group ≤ 8192 := by
    simpa [chunkSum] using weighted_sum_bounds (xs group) (fun i => (ys group i).toInt) (hw group)
  constructor
  · calc
      -(8192 * (count : Int)) = ∑ _group ∈ Finset.range count, (-8192 : Int) := by simp; ring
      _ ≤ blockPrefix xs ys count := Finset.sum_le_sum fun group _ => (bounds group).1
  · calc
      blockPrefix xs ys count ≤ ∑ _group ∈ Finset.range count, (8192 : Int) :=
        Finset.sum_le_sum fun group _ => (bounds group).2
      _ = 8192 * (count : Int) := by simp; ring

/-- A BK32/BK64/BK128 dot starts at zero and contains at most four K32
instructions. Saturation and wrapping are both proved inactive inside it. -/
theorem zeroSeededBlock_exact (kind : Instruction) (xs ys : Nat → Fin 32 → Byte)
    (count : Nat) (hc : count ≤ 4)
    (hw : ∀ group i, -1 ≤ (ys group i).toInt ∧ (ys group i).toInt ≤ 2) :
    foldN (fun group seed => referenceInstruction kind (xs group) (ys group) seed) count 0 =
      BitVec.ofInt 32 (blockPrefix xs ys count) := by
  induction count with
  | zero => simp [foldN, blockPrefix]
  | succ count ih =>
    rw [foldN, ih (by omega)]
    have bounds := blockPrefix_bounds xs ys count hw
    have next := blockPrefix_bounds xs ys (count + 1) hw
    have step : blockPrefix xs ys (count + 1) = blockPrefix xs ys count + chunkSum xs ys count := by
      simp only [blockPrefix, Finset.sum_range_succ]
    rw [step] at next
    rw [referenceInstruction_exact kind (xs count) (ys count) (blockPrefix xs ys count)
      (by constructor <;> omega) (by simpa only [chunkSum] using
        (show -2147483648 ≤ blockPrefix xs ys count + chunkSum xs ys count ∧
          blockPrefix xs ys count + chunkSum xs ys count ≤ 2147483647 from by constructor <;> omega))]
    rw [step]
    rfl

#axiomsClean weighted_product_bounds
#axiomsClean bounded_prefix_representable
#axiomsClean referenceInstruction_exact
#axiomsClean zeroSeededBlock_exact
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked
