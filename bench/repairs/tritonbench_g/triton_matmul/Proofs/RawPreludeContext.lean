import bench.repairs.tritonbench_g.triton_matmul.Proofs.RawPrelude
import bench.repairs.tritonbench_g.triton_matmul.Proofs.RawArithmetic
import VeriTile.Triton.Semantics.Integer

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSource
open VeriTile.Triton FP.Structural FPValue
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

def HasAcc {α : Type} (p : Parameters) (a : CIndex p → α) (s : State α) : Prop :=
  s.regs .real [BM p,BN p] "accumulator" = some a

private theorem int_min_form (a b : Int) : min a b = if a < b then a else b := by
  by_cases h : a < b
  · rw [if_pos h,min_eq_left (by omega)]
  · rw [if_neg h,min_eq_right (by omega)]

/-- Every original metadata value and the actual signed masks survive the
prefix. No natural-index equality or floating operation identity is assumed. -/
theorem preludeContext {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α) :
    Context p s (preludeExecution M D p st s).val ∧
      HasAcc p (fun _ => M.literal (some .fp32) .real 0) (preludeExecution M D p st s).val := by
  have hM : ¬(p.M : Int) < 0 := not_lt_of_ge (Int.natCast_nonneg _)
  have hN : ¬(p.N : Int) < 0 := not_lt_of_ge (Int.natCast_nonneg _)
  have hK : ¬(p.K : Int) < 0 := not_lt_of_ge (Int.natCast_nonneg _)
  have hp : ¬(s.pids 0 : Int) < 0 := not_lt_of_ge (Int.natCast_nonneg _)
  have hr (w : Int) : ¬(s.pids 0 : Int) < w * ((s.pids 0 : Int) / w) := by
    have h := Int.tmod_nonneg w (Int.natCast_nonneg (s.pids 0))
    rw [Int.tmod_def,Int.natCast_tdiv_eq_ediv] at h
    omega
  have hd (w b : Int) :
      ((s.pids 0 : Int) - w * ((s.pids 0 : Int) / w)).tdiv b =
        ((s.pids 0 : Int) - w * ((s.pids 0 : Int) / w)) / b :=
    Int.tdiv_eq_ediv_of_nonneg (by have h := hr w; omega)
  cases st <;> constructor
  all_goals try constructor
  all_goals simp [preludeExecution,backend,HasAcc,control,signedCeil,maskM,maskN,offsetM,offsetN,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,IntegerExpr.sameShape,
    IntegerExpr.tdiv_formula,← Int.tmod_def,int_min_form]
  all_goals try rfl
  all_goals funext i
  all_goals simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    IntegerExpr.sameShape,IntegerExpr.tdiv_formula,Int.tmod_def,int_min_form,hM,hN,hK,hp,hr,hd]

#axiomsClean preludeContext
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSource
