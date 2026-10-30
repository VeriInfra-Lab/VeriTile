import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.IntegerDotPlanBridge
import VeriTile.Triton.Float.IntegerDotIO

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Quote.SharedInteger
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

def context (q : IntegerModel I F H) (rows cols : Nat) : IntegerDot.Context where
  rows := rows
  cols := cols
  rowRegister := "pid_m"
  columnRegister := "pid_n"
  iterationRegister := "k"
  splitAxis := 1
  plans := fun k row col split iteration => plan (q.plans k row col split iteration)

theorem dotAdd_eq (q : IntegerModel I F H) (rows cols : Nat) {α : Type}
    (s : State α) {m k n : Nat} (old : TileIndex [m,n] → Int)
    (a : TileIndex [m,k] → Int) (b : TileIndex [k,n] → Int) :
    IntegerDot.dotAdd (arithmetic q) (context q rows cols) s old a b =
      dotAdd q rows cols s old a b := by
  funext i
  exact plan_eval q _ _ _ _

theorem addExpression_eq {α : Type} [Inhabited α] (M : Algebra α)
    (q : IntegerModel I F H) (rows cols : Nat)
    (bc : Broadcast leftShape rightShape shape) (old : Op .int leftShape)
    (right : Op .int rightShape) (s : State α) :
    IntegerDot.addExpression M (arithmetic q) (context q rows cols) bc old right s =
      addExpression M q rows cols bc old right s := by
  simp only [IntegerDot.addExpression,addExpression,dotAdd_eq]
  rfl

theorem arithmeticStep_eq {α : Type} [Inhabited α] (M : Algebra α)
    (q : IntegerModel I F H) (rows cols : Nat) (statement : ComputeStmt) (s : State α) :
    IntegerDot.arithmeticStep M (arithmetic q) (context q rows cols) statement s =
      arithmeticStep M q rows cols statement s := by
  simp only [IntegerDot.arithmeticStep,arithmeticStep,addExpression_eq]
  rfl

mutual
theorem step_eq {α : Type} [Inhabited α] (M : Algebra α)
    (q : IntegerModel I F H) (rows cols : Nat) (statement : ComputeStmt) (s : State α) :
    IntegerDot.step M (arithmetic q) (context q rows cols) statement s =
      step M q rows cols statement s := by
  cases statement
  case forRange name start stop stride body =>
    simp only [IntegerDot.step_unfold,Quote.step.eq_def]
    exact range_eq M q rows cols name start stop stride body s
  case ifThen cond body =>
    simp only [IntegerDot.step_unfold,Quote.step.eq_def]
    cases h : FP.Structural.evalOp M none cond s with
    | none => rfl
    | some value =>
      change (if value PUnit.unit then _ else _) = _
      split
      · rename_i active
        simpa [active] using run_eq M q rows cols body s
      · rename_i inactive
        simp [inactive]
  case ifThenElse cond yes no =>
    simp only [IntegerDot.step_unfold,Quote.step.eq_def]
    cases h : FP.Structural.evalOp M none cond s with
    | none => rfl
    | some value =>
      change (if value PUnit.unit then _ else _) = _
      split
      · rename_i active
        simpa [active] using run_eq M q rows cols yes s
      · rename_i inactive
        simpa [inactive] using run_eq M q rows cols no s
  all_goals simp only [IntegerDot.step_unfold,Quote.step.eq_def]
  all_goals exact arithmeticStep_eq M q rows cols _ s
termination_by (sizeOf statement,0)
decreasing_by
  all_goals simp_wf
  all_goals try omega
  all_goals have string_positive : ∀ n : RegName, 0 < sizeOf n := by intro n; cases n; simp
  all_goals rename_i idx start stop stride body _ _
  all_goals have hsize := string_positive idx
  all_goals omega

theorem run_eq {α : Type} [Inhabited α] (M : Algebra α)
    (q : IntegerModel I F H) (rows cols : Nat) (code : List ComputeStmt) (s : State α) :
    IntegerDot.run M (arithmetic q) (context q rows cols) code s =
      run M q rows cols code s := by
  cases code with
  | nil => simp only [IntegerDot.run,run]
  | cons statement rest =>
      rw [IntegerDot.run,run]
      rw [step_eq]
      congr 1
      funext t
      exact run_eq M q rows cols rest t
termination_by (sizeOf code,0)
decreasing_by all_goals (simp_wf; omega)

theorem range_eq {α : Type} [Inhabited α] (M : Algebra α)
    (q : IntegerModel I F H) (rows cols : Nat)
    (name : RegName) (start stop stride : Nat) (body : List ComputeStmt) (s : State α) :
    IntegerDot.range M (arithmetic q) (context q rows cols) name start stop stride body s =
      range M q rows cols name start stop stride body s := by
  rw [IntegerDot.range,range]
  split
  · rw [run_eq]
    congr 1
    funext t
    exact range_eq M q rows cols name (start+stride) stop stride body t
  · rfl
termination_by (sizeOf body+1,stop-start)
decreasing_by all_goals omega
end

theorem exec_eq {α : Type} [Inhabited α] (M : Algebra α)
    (q : IntegerModel I F H) (rows cols : Nat) (source : ComputeKernel) (s : State α) :
    IntegerDot.exec M (arithmetic q) (context q rows cols) source s =
      execute M q rows cols source s := run_eq M q rows cols source.surfaceBody s

#axiomsClean dotAdd_eq
#axiomsClean addExpression_eq
#axiomsClean arithmeticStep_eq
#axiomsClean exec_eq
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Quote.SharedInteger
