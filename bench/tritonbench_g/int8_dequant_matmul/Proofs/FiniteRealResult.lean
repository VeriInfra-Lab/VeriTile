import bench.tritonbench_g.int8_dequant_matmul.Mathematical
import bench.tritonbench_g.int8_dequant_matmul.Proofs.QuoteModel

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.FiniteRealResult
noncomputable section
set_option maxHeartbeats 4000000

/-- The selected finite row/column, extended by zero solely to reuse the
existing split-partition theorem on natural-indexed mathematical arrays. -/
def slice (s : Shape) (hasBias : Bool) (input : Mathematical.Inputs s ℝ)
    (cell : MatrixCell s) : RealInput s where
  A := fun _ k => if h : k < s.K then ((input.a cell.1 ⟨k,h⟩).toInt : ℝ) else 0
  B := fun k _ => if h : k < s.K then ((input.b ⟨k,h⟩ cell.2).toInt : ℝ) else 0
  x := fun _ => input.rowScale cell.1
  w := fun _ => input.columnScale cell.2
  bias := if hasBias then some (fun _ => input.bias cell.2) else none

theorem operand (s : Shape) (c : Config) (hasBias : Bool)
    (input : Mathematical.Inputs s ℝ) (cell : MatrixCell s) (t z : Nat) (lane : Fin c.BK) :
    realProduct (Mathematical.left s c input cell.1 t z lane)
      (Mathematical.right s c input cell.2 t z lane) =
      if sourceK c t z lane.val < s.K then
        (slice s hasBias input cell).A cell.1.val (sourceK c t z lane.val) *
        (slice s hasBias input cell).B (sourceK c t z lane.val) cell.2.val else 0 := by
  by_cases active : sourceK c t z lane.val < s.K <;>
    simp [Mathematical.left,Mathematical.right,realProduct,slice,active]

theorem accumulator (s : Shape) (c : Config) (selected : Selection s c)
    (hasBias : Bool) (input : Mathematical.Inputs s ℝ) (cell : MatrixCell s)
    (split : Fin c.S) (t : Nat) :
    ((Mathematical.accumulator Quote.integerRealOps s c selected input cell split t : Int) : ℝ) =
      rsum (fun t => rsum (fun lane => if sourceK c t split.val lane < s.K then
        (slice s hasBias input cell).A cell.1.val (sourceK c t split.val lane) *
        (slice s hasBias input cell).B (sourceK c t split.val lane) cell.2.val else 0) c.BK) t := by
  induction t with
  | zero => simp [Mathematical.accumulator,Quote.integerRealOps,rsum]
  | succ t ih =>
      rw [Mathematical.accumulator,Quote.integer_plan_real,DotPlan.real,ih]
      simp_rw [operand s c hasBias input cell t split.val]
      rw [finRange_rsum (fun lane => if sourceK c t split.val lane < s.K then
        (slice s hasBias input cell).A cell.1.val (sourceK c t split.val lane) *
        (slice s hasBias input cell).B (sourceK c t split.val lane) cell.2.val else 0) c.BK]
      rfl

theorem split_value (s : Shape) (c : Config) (selected : Selection s c)
    (hasBias : Bool) (input : Mathematical.Inputs s ℝ) (cell : MatrixCell s)
    (split : Fin c.S) :
    Mathematical.splitValue Quote.integerRealOps s c selected (1/16129) hasBias input cell split =
      realPartial s c (slice s hasBias input cell) cell.1.val cell.2.val split.val := by
  unfold Mathematical.splitValue
  simp only [Quote.integerRealOps,id_eq]
  have acc := accumulator s c selected hasBias input cell split (steps s c)
  simp only [Quote.integerRealOps] at acc
  rw [acc]
  cases hasBias <;> by_cases zero : split.val = 0 <;>
    simp [realPartial,slice,biasAt,zero,div_eq_mul_inv]

theorem target_eq (s : Shape) (hasBias : Bool) (input : Mathematical.Inputs s ℝ)
    (cell : MatrixCell s) :
    target s (slice s hasBias input cell) cell.1.val cell.2.val =
      Mathematical.real s hasBias input cell := by
  have sum : (List.ofFn (fun k : Fin s.K => ((input.a cell.1 k).toInt : ℝ) *
      ((input.b k cell.2).toInt : ℝ))).sum =
      rsum (fun k => (slice s hasBias input cell).A cell.1.val k *
        (slice s hasBias input cell).B k cell.2.val) s.K := by
    rw [List.ofFn_eq_map]
    simpa only [slice,dif_pos,Fin.isLt] using
      finRange_rsum (fun k => (slice s hasBias input cell).A cell.1.val k *
        (slice s hasBias input cell).B k cell.2.val) s.K
  unfold Mathematical.real
  rw [sum]
  cases hasBias <;> rfl

/-- The existing exact split partition theorem, applied to every cell's own
complete schedule. No order law for floating arithmetic is being assumed. -/
theorem schedule_target (s : Shape) (c : Config) (input : RealInput s)
    (schedule : Schedule s c) (cell : MatrixCell s) :
    cellFold Quote.integerRealOps c
      (fun z => realPartial s c input cell.1.val cell.2.val z.val)
      (projection cell schedule.events) 0 = target s input cell.1.val cell.2.val := by
  let f : Fin c.S → ℝ := fun z => realPartial s c input cell.1.val cell.2.val z.val
  have perm := ((schedule.complete cell).map f).sum_eq
  unfold cellFold
  change (projection cell schedule.events).foldl
    (fun old z => if c.S=1 then f z else old+f z) 0 = _
  by_cases single : c.S=1
  · have len : (projection cell schedule.events).length=1 := by
      simpa [single] using (schedule.complete cell).length_eq
    have singleton : ∃ z,projection cell schedule.events=[z] := by
      cases h : projection cell schedule.events with
      | nil => simp [h] at len
      | cons z zs => cases zs with
        | nil => exact ⟨z,rfl⟩
        | cons z' zs => simp [h] at len
    obtain ⟨z,hz⟩ := singleton
    rw [hz,List.foldl_cons,List.foldl_nil,if_pos single]
    have zero : z.val=0 := by have h := z.isLt; omega
    simpa [single,rsum,f,zero] using partials_target s c input cell.1.val cell.2.val
  · simp only [if_neg single]
    rw [fold_add,zero_add,perm]
    change ((List.finRange c.S).map
      (fun z => realPartial s c input cell.1.val cell.2.val z.val)).sum = _
    rw [finRange_rsum]
    exact partials_target _ _ _ _ _

theorem real (s : Shape) (c : Config) (selected : Selection s c)
    (hasBias : Bool) (schedule : Schedule s c) (input : Mathematical.Inputs s ℝ)
    (cell : MatrixCell s) :
    cellFold Quote.integerRealOps c
      (Mathematical.splitValue Quote.integerRealOps s c selected (1/16129) hasBias input cell)
      (projection cell schedule.events) 0 = Mathematical.real s hasBias input cell := by
  have values : Mathematical.splitValue Quote.integerRealOps s c selected (1/16129) hasBias input cell =
      fun z => realPartial s c (slice s hasBias input cell) cell.1.val cell.2.val z.val := by
    funext z
    exact split_value s c selected hasBias input cell z
  rw [values,schedule_target,target_eq]

#axiomsClean accumulator
#axiomsClean split_value
#axiomsClean real
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.FiniteRealResult
