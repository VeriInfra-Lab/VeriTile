import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Kernels
import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Proofs.IOComplete
import VeriTile.Triton.Float.StructuralSafety
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.PairConnection
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {A : Type} [Inhabited A]

/-- Whole-state execution and actual access-trace preservation, including
failures. This helper is used only to connect the two quoted programs. -/
structure CodeEq (M : Algebra A) (left right : List ComputeStmt) : Prop where
  run : ∀ s, FP.Structural.run M left s = FP.Structural.run M right s
  safe : ∀ bounds s, Safety.traceSafeList M bounds left s ↔ Safety.traceSafeList M bounds right s

structure StepEq (M : Algebra A) (left right : ComputeStmt) : Prop where
  step : ∀ s, FP.Structural.step M left s = FP.Structural.step M right s
  safe : ∀ bounds s, Safety.stepSafe M bounds left s ↔ Safety.stepSafe M bounds right s

theorem code_refl (M : Algebra A) (code : List ComputeStmt) : CodeEq M code code := ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
theorem step_refl (M : Algebra A) (st : ComputeStmt) : StepEq M st st := ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
theorem code_cons {M : Algebra A} {l r : ComputeStmt} {ls rs : List ComputeStmt}
    (head : StepEq M l r) (tail : CodeEq M ls rs) : CodeEq M (l::ls) (r::rs) := by
  constructor
  · intro s
    simp only [run,head.step]
    cases step M r s with
    | none => rfl
    | some t => exact tail.run t
  · intro bounds s
    simp only [Safety.traceSafeList_cons,head.safe,head.step]
    exact and_congr_right (fun _ => forall_congr' (fun t => imp_congr_right (fun _ => tail.safe bounds t)))

theorem step_if {M : Algebra A} (cond : Op .bool []) {l r : List ComputeStmt}
    (body : CodeEq M l r) : StepEq M (.ifThen cond l) (.ifThen cond r) := by
  constructor
  · intro s
    simp only [step]
    cases evalOp M none cond s with
    | none => rfl
    | some v =>
      simp only [bind,Option.bind_some]
      split
      · exact body.run s
      · rfl
  · intro bounds s
    simp only [Safety.stepSafe_unfold]
    cases evalOp M none cond s with
    | none => rfl
    | some v => split <;> simp [body.safe]

theorem step_if_else {M : Algebra A} (cond : Op .bool []) {ly ry ln rn : List ComputeStmt}
    (yes : CodeEq M ly ry) (no : CodeEq M ln rn) :
    StepEq M (.ifThenElse cond ly ln) (.ifThenElse cond ry rn) := by
  constructor
  · intro s
    simp only [step]
    cases evalOp M none cond s with
    | none => rfl
    | some v =>
      simp only [bind,Option.bind_some]
      split
      · exact yes.run s
      · exact no.run s
  · intro bounds s
    simp only [Safety.stepSafe_unfold]
    cases evalOp M none cond s with
    | none => rfl
    | some v => split <;> simp [yes.safe,no.safe]

/-- Exact natural remainder strength reduction. The product subtracted
is at most the dividend; the identity also covers divisor zero. -/
theorem remainder (a b : Nat) : a % b=a-(a/b)*b := Nat.mod_eq_sub_div_mul

theorem no_underflow (a b : Nat) : (a/b)*b≤a := Nat.div_mul_le_self a b

theorem step_remainder (M : Algebra A) (out dividend divisor : RegName) :
    StepEq M
      (.assign .nat [] out (.alg (.tritonMod .nat .nil
        (.ref .nat [] dividend) (.ref .nat [] divisor))))
      (.assign .nat [] out (.alg (.sub .nat .nil (.ref .nat [] dividend)
        (.mul .nat .nil
          (.tritonDiv .nat .nil (.ref .nat [] dividend) (.ref .nat [] divisor))
          (.ref .nat [] divisor))))) := by
  constructor
  · intro s
    simp only [step,evalExpr,evalOp_unfold]
    cases ha : s.regs .nat [] dividend <;> cases hb : s.regs .nat [] divisor
    all_goals simp [ha,hb,bind,Option.bind,numeric,integralMod,integralFloorDiv,bop,←remainder]
  · intro bounds s
    simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeOp_unfold]

/-- Complete original and integer-CSE source equality, including all current
registers, source failures, access traces and memory. No FP numerical law. -/
theorem source_pair (M : Algebra A) (p : Source.Parameters) :
    CodeEq M (Source.kernel p).surfaceBody (Source.quotientRemainder p).surfaceBody := by
  unfold Source.kernel Source.quotientRemainder
  cases dm : p.divM <;> cases dn : p.divN <;> cases dk : p.divK
  all_goals
    simp only [dm,dn,dk]
    unfold ComputeKernel.surfaceBody
    repeat' first
      | exact code_refl M _
      | exact step_refl M _
      | apply code_cons
      | apply step_if
      | apply step_if_else
      | exact step_remainder M _ _ _

#axiomsClean source_pair
end
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.PairConnection
