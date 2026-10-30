import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.Memory
namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton
set_option autoImplicit false
set_option maxHeartbeats 2000000
noncomputable section

/-- Independent maximum over exactly the logical row coordinates. -/
def logicalMaximum (values : Nat → ℝ) : Nat → ℝ
  | 0 => 0
  | n + 1 => max (logicalMaximum values n) |values n|

theorem logicalMaximum_nonnegative (values : Nat → ℝ) (n : Nat) : 0 ≤ logicalMaximum values n := by
  induction n with
  | zero => exact le_rfl
  | succ n ih => exact le_trans ih (le_max_left _ _)

theorem logicalMaximum_bound (values : Nat → ℝ) (n i : Nat) (hi : i < n) :
    |values i| ≤ logicalMaximum values n := by
  induction n with
  | zero => omega
  | succ n ih =>
    by_cases he : i = n
    · subst i; exact le_max_right _ _
    · exact le_trans (ih (by omega)) (le_max_left _ _)

theorem logicalMaximum_least (values : Nat → ℝ) (n : Nat) (bound : ℝ)
    (zero : 0 ≤ bound) (upper : ∀ i, i < n → |values i| ≤ bound) :
    logicalMaximum values n ≤ bound := by
  induction n with
  | zero => exact zero
  | succ n ih =>
    exact max_le (ih (fun i hi => upper i (by omega))) (upper n (by omega))

theorem logicalMaximum_zero (values : Nat → ℝ) (n : Nat) :
    logicalMaximum values n = 0 ↔ ∀ i, i < n → values i = 0 := by
  constructor
  · intro hm i hi
    have h := logicalMaximum_bound values n i hi
    rw [hm] at h
    exact abs_eq_zero.mp (le_antisymm h (abs_nonneg _))
  · intro zero
    apply le_antisymm
    · apply logicalMaximum_least values n 0 le_rfl
      intro i hi
      simp [zero i hi]
    · exact logicalMaximum_nonnegative values n

theorem Tree.maximum_upper (tree : Tree) (values : Nat → ℝ) (bound : ℝ)
    (upper : ∀ i ∈ tree.leaves, values i ≤ bound) : tree.evaluate max values ≤ bound := by
  induction tree with
  | leaf i => exact upper i (by simp [Tree.leaves])
  | node a b ha hb =>
    exact max_le (ha (fun i hi => upper i (by simp [Tree.leaves, hi])))
      (hb (fun i hi => upper i (by simp [Tree.leaves, hi])))

theorem Tree.maximum_member (tree : Tree) (values : Nat → ℝ) (i : Nat) (hi : i ∈ tree.leaves) :
    values i ≤ tree.evaluate max values := by
  induction tree with
  | leaf j =>
    simp only [Tree.leaves, List.mem_singleton] at hi
    subst i
    exact le_rfl
  | node a b ha hb =>
    rcases List.mem_append.mp hi with h | h
    · exact le_trans (ha h) (le_max_left _ _)
    · exact le_trans (hb h) (le_max_right _ _)

/-- Nearest integer with ties to the even integer, independently of libdevice. -/
def roundEven (x : ℝ) : Int :=
  let lower := Int.floor x
  if x - lower < 1 / 2 then lower
  else if 1 / 2 < x - lower then lower + 1
  else if lower % 2 = 0 then lower else lower + 1

@[simp] theorem roundEven_zero : roundEven 0 = 0 := by norm_num [roundEven]

def idealOperations (half : ℝ → Half) : Operations ℝ where
  zero := 0
  one := 1
  scale := 127
  abs := abs
  maximum := max
  isZero := fun x => decide (x = 0)
  divide := (· / ·)
  multiply := (· * ·)
  llrint := roundEven
  toHalf := half

theorem source_maximum_is_logical (half : ℝ → Half) (p : Launch) (values : Nat → ℝ) :
    rowMaximum (idealOperations half) p values = logicalMaximum values p.columns := by
  let lanes := fun i => if i < p.columns then |values i| else (0 : ℝ)
  change p.tree.evaluate max lanes = logicalMaximum values p.columns
  apply le_antisymm
  · apply Tree.maximum_upper
    intro i _
    by_cases hi : i < p.columns
    · simpa [lanes, hi] using logicalMaximum_bound values p.columns i hi
    · simpa [lanes, hi] using logicalMaximum_nonnegative values p.columns
  · apply logicalMaximum_least
    · have hz : 0 ∈ p.tree.leaves := p.leafCover.mem_iff.mpr (List.mem_range.mpr (lt_of_lt_of_le p.positive p.covers))
      have h := Tree.maximum_member p.tree lanes 0 hz
      have h0 : 0 ≤ lanes 0 := by simp [lanes, p.positive, abs_nonneg]
      exact le_trans h0 h
    · intro i hi
      have hm : i ∈ p.tree.leaves := p.leafCover.mem_iff.mpr (List.mem_range.mpr (lt_of_lt_of_le hi p.covers))
      simpa [lanes, hi] using Tree.maximum_member p.tree lanes i hm

def mathematicalByte (values : Nat → ℝ) (columns column : Nat) : Byte :=
  let maximum := logicalMaximum values columns
  if maximum = 0 then narrow8 0 else narrow8 (roundEven (127 * (values column / maximum)))

theorem source_quantization_is_logical (half : ℝ → Half) (p : Launch) (values : Nat → ℝ)
    (column : Nat) (hc : column < p.columns) :
    rowQuantized (idealOperations half) (values column) (rowMaximum (idealOperations half) p values) =
      mathematicalByte values p.columns column := by
  rw [source_maximum_is_logical]
  by_cases hm : logicalMaximum values p.columns = 0
  · have hz := (logicalMaximum_zero values p.columns).mp hm column hc
    simp [rowQuantized, rowScaled, idealOperations, mathematicalByte, hm, hz]
  · simp [rowQuantized, rowScaled, idealOperations, mathematicalByte, hm]

theorem zero_row_quantization (half : ℝ → Half) (p : Launch) (values : Nat → ℝ)
    (zero : ∀ i, i < p.columns → values i = 0) (column : Nat) (hc : column < p.columns) :
    rowQuantized (idealOperations half) (values column) (rowMaximum (idealOperations half) p values) = narrow8 0 := by
  rw [source_quantization_is_logical half p values column hc]
  simp [mathematicalByte, (logicalMaximum_zero values p.columns).mpr zero]

#axiomsClean source_maximum_is_logical
#axiomsClean source_quantization_is_logical
#axiomsClean zero_row_quantization
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton
