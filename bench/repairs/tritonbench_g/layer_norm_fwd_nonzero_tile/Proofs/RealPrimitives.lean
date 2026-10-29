import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.RealModel

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.ActualMathematics
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
noncomputable section
open scoped BigOperators Classical
set_option maxHeartbeats 3000000
set_option maxRecDepth 10000

@[simp] theorem literal (x : ℝ) : TypedSource.lit algebra x = x := rfl
@[simp] theorem cast (x : ℝ) : TypedSource.cast algebra x = x := rfl
@[simp] theorem op_add (x y : ℝ) : TypedSource.op algebra .add x y = x + y := rfl
@[simp] theorem op_sub (x y : ℝ) : TypedSource.op algebra .sub x y = x - y := rfl
@[simp] theorem op_mul (x y : ℝ) : TypedSource.op algebra .mul x y = x * y := rfl
@[simp] theorem op_div (x y : ℝ) : TypedSource.op algebra .div x y = x / y := rfl
@[simp] theorem default_add (x y : ℝ) : TypedSource.defaultOp algebra .add x y = x + y := rfl
@[simp] theorem default_sub (x y : ℝ) : TypedSource.defaultOp algebra .sub x y = x - y := rfl
@[simp] theorem default_mul (x y : ℝ) : TypedSource.defaultOp algebra .mul x y = x * y := rfl

@[simp] theorem count (N : Nat) : TypedSource.countValue algebra N = (N : ℝ) := by
  by_cases h : N=1 <;> simp [TypedSource.countValue,TypedSource.lit,h,algebra,engine,Algebra.withDefaultPrecision]

theorem random (seed : Int) (col : Nat) :
    TypedSource.randomValue algebra seed col =
      Random.randReal (Formula.seedU32 seed) (col : Int) .bits32 7 := by
  have nonnegative : 0 ≤ seed % (2^32 : Int) := Int.emod_nonneg _ (by norm_num)
  change (((Random.rand ((seed % (2^32 : Int)).toNat : Int) (col : Int) .bits32 7).toRat?).getD 0 : ℝ) = _
  rw [Int.toNat_of_nonneg nonnegative,Random.rand_toRat_randRat]
  rfl

@[simp] theorem keep (seed : Int) (p : ℝ) (col : Nat) :
    TypedSource.keepValue algebra comparisons seed p col = Formula.keep seed p col := by
  change decide (p < TypedSource.randomValue algebra seed col) = _
  rw [random]
  rfl

/-- Masked tile reduction equals the independent logical N-column sum,
including the empty column domain. -/
theorem sum_prefix (N B : Nat) (coverage : N ≤ B) (f : Nat → ℝ) :
    (∑ j : Fin B, if j.val < N then f j.val else 0) = ∑ j ∈ Finset.range N, f j := by
  rw [Fin.sum_univ_eq_sum_range (fun j => if j < N then f j else 0) B]
  calc
    (∑ j ∈ Finset.range B, if j < N then f j else 0) =
        ∑ j ∈ Finset.range N, if j < N then f j else 0 := by
      symm
      apply Finset.sum_subset (Finset.range_mono coverage)
      intro j _ hj
      simp only [Finset.mem_range] at hj
      simp [hj]
    _ = _ := Finset.sum_congr rfl (fun j hj => if_pos (Finset.mem_range.mp hj))

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.ActualMathematics
