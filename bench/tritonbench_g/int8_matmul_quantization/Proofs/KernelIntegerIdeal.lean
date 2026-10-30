import bench.tritonbench_g.int8_matmul_quantization.Proofs.Ideal
import bench.tritonbench_g.int8_matmul_quantization.Proofs.KernelExecution

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelIntegerIdeal
open Int8DequantMatmul
noncomputable section
set_option maxHeartbeats 4000000

/-- Exact mathematical integers for the Real interpretation. The native
interpretation is separately signed32 and preserves instruction saturation. -/
def operations : Int8MatmulQuantization.Ops Int ℝ ℝ where
  izero := 0
  fzero := 0
  hzero := 0
  product8 := fun x y => x.toInt*y.toInt
  iadd := (· + ·)
  mma32 := fun _ xs ys old => old + (List.ofFn (fun i => (xs i).toInt*(ys i).toInt)).sum
  toFloat := fun x => (x : ℝ)
  mul := (· * ·)
  narrow := id
  hadd := (· + ·)
  widen := id
  add := (· + ·)
  absolute := abs
  maximum := max
  divide := (· / ·)
  scale127 := fun x => x/127
  integer8 := realInteger8

private theorem option_sum (xs : List (Option α)) (f : α → Int) :
    ((xs.filterMap id).map f).sum = (xs.map (fun x => (x.map f).getD 0)).sum := by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    cases x <;> simpa only [List.filterMap_cons, id_eq, Option.map_none, Option.map_some,
      Option.getD_none, Option.getD_some, List.map_cons, List.sum_cons, zero_add,
      add_right_inj] using ih

theorem micro_integer (op : Micro BK) (xs ys : Fin BK → BitVec 8) (old : Int) :
    op.eval operations.toOps xs ys old =
      old + (op.terms.map (fun k => (xs k).toInt*(ys k).toInt)).sum := by
  cases op with
  | scalar k => simp [Micro.eval, Micro.terms, operations]
  | instruction kind lanes =>
    simp only [Micro.eval, Micro.terms, operations, option_sum, List.map_ofFn]
    congr 2
    congr 1
    funext i
    cases h : lanes i <;> simp [h]

theorem operations_integer (instructions : List (Micro BK)) (xs ys : Fin BK → BitVec 8) (old : Int) :
    instructions.foldl (fun v op => op.eval operations.toOps xs ys v) old =
      old + ((instructions.flatMap Micro.terms).map (fun k => (xs k).toInt*(ys k).toInt)).sum := by
  induction instructions generalizing old with
  | nil => simp
  | cons op instructions ih =>
    rw [List.foldl_cons, ih, micro_integer]
    simp only [List.flatMap_cons, List.map_append, List.sum_append]
    ring

theorem plan_integer (plan : DotPlan BK) (xs ys : Fin BK → BitVec 8) (old : Int) :
    plan.eval operations.toOps xs ys old =
      old + ((List.finRange BK).map (fun k => (xs k).toInt*(ys k).toInt)).sum := by
  have h := (plan.complete.map (fun k => (xs k).toInt*(ys k).toInt)).sum_eq
  unfold Int8DequantMatmul.DotPlan.eval
  split_ifs
  · rw [operations_integer, h]
  · change old + _ = _
    rw [operations_integer, h]
    simp [operations]

/-- Reuses the established mathematical DotPlan result without a native
overflow assumption. This is a cast of exact integer sums only. -/
theorem plan_cast_real (plan : DotPlan BK) (xs ys : Fin BK → BitVec 8) (old : Int) :
    ((plan.eval operations.toOps xs ys old : Int) : ℝ) =
      plan.eval Int8DequantMatmul.realOps xs ys (old : ℝ) := by
  rw [plan_integer, Int8DequantMatmul.DotPlan.real]
  simp [Int8DequantMatmul.realProduct, Int.cast_list_sum, Int.cast_add, Int.cast_mul,
    List.map_map, Function.comp_def]

#axiomsClean plan_integer
#axiomsClean plan_cast_real
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelIntegerIdeal
