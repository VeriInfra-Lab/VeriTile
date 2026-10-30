import VeriTile.Triton.Float.IntegerDot
import VeriTile.Triton.Float.StructuralSafety

/-! Access safety follows the concrete IntegerDot execution, including its
actual integer accumulator successors. The fused add/dot changes arithmetic
only: all of its source children retain Structural expression safety. Other
control forms execute in the unchanged Structural interpreter and use its
matching checker. Unsupported effects continue to fail closed. -/
namespace VeriTile.Triton.FP.Structural.IntegerDot
noncomputable section

mutual
def stepSafe {α : Type} [Inhabited α] (M : Algebra α)
    (arithmetic : Arithmetic) (context : Context) (bounds : RegionName → Nat) :
    ComputeStmt → State α → Prop
  | .forRange name start stop stride body, s =>
      rangeSafe M arithmetic context bounds name start stop stride body s
  | .ifThenElse cond yes no, s =>
      Safety.safeOp M none bounds s cond ∧
      match evalOp M none cond s with
      | some c => if c PUnit.unit then traceSafeList M arithmetic context bounds yes s
          else traceSafeList M arithmetic context bounds no s
      | none => True
  | .ifThen cond body, s =>
      match context.mode with
      | .scalarCast=>
          Safety.safeOp M none bounds s cond ∧
          match evalOp M none cond s with
          | some c => if c PUnit.unit then traceSafeList M arithmetic context bounds body s else True
          | none => True
      | .accumulatorNat=>Safety.stepSafe M bounds (.ifThen cond body) s
  | statement, s => Safety.stepSafe M bounds statement s
termination_by statement _ => (sizeOf statement, 0)
decreasing_by
  all_goals simp_wf
  all_goals try omega
  all_goals have : 0 < sizeOf name := by cases name; simp
  all_goals omega

def traceSafeList {α : Type} [Inhabited α] (M : Algebra α)
    (arithmetic : Arithmetic) (context : Context) (bounds : RegionName → Nat) :
    List ComputeStmt → State α → Prop
  | [], _ => True
  | statement :: rest, s => stepSafe M arithmetic context bounds statement s ∧
      match step M arithmetic context statement s with
      | some t => traceSafeList M arithmetic context bounds rest t
      | none => True
termination_by code _ => (sizeOf code, 0)
decreasing_by all_goals (simp_wf; omega)

def rangeSafe {α : Type} [Inhabited α] (M : Algebra α)
    (arithmetic : Arithmetic) (context : Context) (bounds : RegionName → Nat)
    (name : RegName) (start stop stride : Nat) (body : List ComputeStmt)
    (s : State α) : Prop :=
  if 0 < stride ∧ start < stop then
    traceSafeList M arithmetic context bounds body (s.setReg name .nat [] (fun _ => start)) ∧
    match run M arithmetic context body (s.setReg name .nat [] (fun _ => start)) with
    | some t => rangeSafe M arithmetic context bounds name (start + stride) stop stride body t
    | none => True
  else True
termination_by (sizeOf body + 1, stop - start)
decreasing_by all_goals omega
end

def TraceSafe {α : Type} [Inhabited α] (M : Algebra α)
    (arithmetic : Arithmetic) (context : Context) (bounds : RegionName → Nat)
    (source : ComputeKernel) (s : State α) : Prop :=
  traceSafeList M arithmetic context bounds source.surfaceBody s

abbrev stepSafe_unfold := @stepSafe.eq_def
abbrev traceSafeList_unfold := @traceSafeList.eq_def
abbrev rangeSafe_unfold := @rangeSafe.eq_def

@[simp] theorem traceSafeList_nil {α : Type} [Inhabited α] (M : Algebra α)
    (a : Arithmetic) (c : Context) (bounds : RegionName → Nat) (s : State α) :
    traceSafeList M a c bounds [] s := by rw [traceSafeList]; trivial

theorem traceSafeList_cons {α : Type} [Inhabited α] (M : Algebra α)
    (a : Arithmetic) (c : Context) (bounds : RegionName → Nat)
    (st : ComputeStmt) (rest : List ComputeStmt) (s : State α) :
    traceSafeList M a c bounds (st :: rest) s ↔ stepSafe M a c bounds st s ∧
      ∀ t, step M a c st s = some t → traceSafeList M a c bounds rest t := by
  simp only [traceSafeList]
  cases h : step M a c st s <;> simp

theorem traceSafeList_append {α : Type} [Inhabited α] (M : Algebra α)
    (a : Arithmetic) (c : Context) (bounds : RegionName → Nat)
    (before after : List ComputeStmt) (s : State α) :
    traceSafeList M a c bounds (before ++ after) s ↔
      traceSafeList M a c bounds before s ∧
      ∀ t, run M a c before s = some t → traceSafeList M a c bounds after t := by
  induction before generalizing s with
  | nil => simp [run]
  | cons st rest ih =>
      cases h : step M a c st s <;>
        simp [traceSafeList, run, h, ih, and_assoc]

end
end VeriTile.Triton.FP.Structural.IntegerDot
