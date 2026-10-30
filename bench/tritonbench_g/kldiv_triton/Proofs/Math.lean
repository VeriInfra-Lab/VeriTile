import bench.tritonbench_g.kldiv_triton.Proofs.Mathematical_Math

namespace VeriTile.Bench.Repairs.KldivTriton.Math
open scoped BigOperators
set_option autoImplicit false
noncomputable section

inductive Mode where
  | none | sum | mean | batchmean
  deriving DecidableEq

/-- Independent logical KL term. The probability-zero extension is explicit;
the Real formula is total and makes no assertion about native NaN arithmetic. -/
def loss (logTarget : Bool) (prediction target : ℝ) : ℝ :=
  if logTarget then Real.exp target * (target - prediction)
  else if target = 0 then 0 else target * (Real.log target - prediction)

def gradient (logTarget : Bool) (target : ℝ) : ℝ :=
  if logTarget then -Real.exp target else -target

def total (rows cols : Nat) (flag : Bool) (prediction target : Nat → Nat → ℝ) : ℝ :=
  ∑ r : Fin rows, ∑ c : Fin cols, loss flag (prediction r.val c.val) (target r.val c.val)

def reduced (mode : Mode) (rows cols : Nat) (flag : Bool)
    (prediction target : Nat → Nat → ℝ) : ℝ :=
  let value := total rows cols flag prediction target
  match mode with
  | .none | .sum => value
  | .mean => value / (rows * cols : Nat)
  | .batchmean => value / rows

theorem loss_original (flag : Bool) (prediction target : ℝ) :
    loss flag prediction target =
      VeriTile.Bench.TritonBenchG.KldivTriton.FullMath.loss flag prediction target := by
  cases flag <;> by_cases zero : target = 0 <;>
    simp [loss, VeriTile.Bench.TritonBenchG.KldivTriton.FullMath.loss, zero]

theorem gradient_original (flag : Bool) (target : ℝ) :
    gradient flag target = VeriTile.Bench.TritonBenchG.KldivTriton.FullMath.gradient flag target := rfl

theorem raw_zero (prediction : ℝ) : loss Bool.false prediction 0 = 0 := by simp [loss]

theorem empty_total (cols : Nat) (flag : Bool) (prediction target : Nat → Nat → ℝ) :
    total 0 cols flag prediction target = 0 := by simp [total]

theorem empty_columns (rows : Nat) (flag : Bool) (prediction target : Nat → Nat → ℝ) :
    total rows 0 flag prediction target = 0 := by simp [total]

theorem sum_ranges_mul (chunks block : Nat) (f : Nat → ℝ) :
    (∑ k ∈ Finset.range chunks, ∑ j ∈ Finset.range block, f (k * block + j)) =
      ∑ i ∈ Finset.range (chunks * block), f i := by
  induction chunks with
  | zero => simp
  | succ k ih =>
    rw [Finset.sum_range_succ, ih, Nat.succ_mul, Finset.sum_range_add]

theorem padded_sum (cols capacity : Nat) (h : cols ≤ capacity) (f : Nat → ℝ) :
    (∑ j ∈ Finset.range capacity, if j < cols then f j else 0) =
      ∑ j ∈ Finset.range cols, f j := by
  rw [show capacity = cols + (capacity - cols) by omega, Finset.sum_range_add]
  have first : (∑ j ∈ Finset.range cols, if j < cols then f j else 0) =
      ∑ j ∈ Finset.range cols, f j := by
    apply Finset.sum_congr rfl
    intro j hj
    exact if_pos (Finset.mem_range.mp hj)
  have tail : (∑ j ∈ Finset.range (capacity - cols),
      if cols + j < cols then f (cols + j) else 0) = 0 := by
    apply Finset.sum_eq_zero
    intro j _
    exact if_neg (by omega)
  rw [first, tail, add_zero]

/-- Summing the actual per-chunk outputs covers every logical column once.
This is a Real theorem; native sums retain their separate addition schedules. -/
theorem chunks_total (cols block : Nat) (hBlock : 0 < block) (flag : Bool)
    (prediction target : Nat → ℝ) :
    (∑ k : Fin (VeriTile.Bench.TritonBenchG.KldivTriton.FullMath.rounds cols block),
      VeriTile.Bench.TritonBenchG.KldivTriton.FullMath.chunkLoss
        flag prediction target cols block k.val) =
      ∑ c : Fin cols, loss flag (prediction c.val) (target c.val) := by
  simp only [VeriTile.Bench.TritonBenchG.KldivTriton.FullMath.chunkLoss,
    VeriTile.Bench.TritonBenchG.KldivTriton.FullMath.paddedLoss_eq,
    ← loss_original]
  rw [Fin.sum_univ_eq_sum_range (fun k => ∑ j : Fin block,
    if k * block + j.val < cols then
      loss flag (prediction (k * block + j.val)) (target (k * block + j.val)) else 0)]
  have inner (k : Nat) : (∑ j : Fin block, if k * block + j.val < cols then
      loss flag (prediction (k * block + j.val)) (target (k * block + j.val)) else 0) =
      ∑ j ∈ Finset.range block, if k * block + j < cols then
        loss flag (prediction (k * block + j)) (target (k * block + j)) else 0 :=
    Fin.sum_univ_eq_sum_range (fun j => if k * block + j < cols then
      loss flag (prediction (k * block + j)) (target (k * block + j)) else 0) block
  simp only [inner]
  rw [sum_ranges_mul _ block (fun c => if c < cols then loss flag (prediction c) (target c) else 0)]
  rw [Fin.sum_univ_eq_sum_range (fun c => loss flag (prediction c) (target c))]
  exact padded_sum cols _
    (VeriTile.Bench.TritonBenchG.KldivTriton.FullMath.rounds_cover cols block hBlock) _

theorem row_chunks_total (rows cols block : Nat) (hBlock : 0 < block) (flag : Bool)
    (prediction target : Nat → Nat → ℝ) :
    (∑ r : Fin rows,
      ∑ k : Fin (VeriTile.Bench.TritonBenchG.KldivTriton.FullMath.rounds cols block),
        VeriTile.Bench.TritonBenchG.KldivTriton.FullMath.chunkLoss
          flag (prediction r.val) (target r.val) cols block k.val) =
      total rows cols flag prediction target := by
  unfold total
  apply Finset.sum_congr rfl
  intro r _
  exact chunks_total cols block hBlock flag _ _

#specNonCircular loss avoiding [VeriTile.Bench.TritonBenchG.KldivTriton.FullMath.loss]
#axiomsClean loss_original
#axiomsClean gradient_original
#axiomsClean raw_zero
#axiomsClean empty_total
#axiomsClean empty_columns
#axiomsClean chunks_total
#axiomsClean row_chunks_total
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Math
