import bench.repairs.tritonbench_g.l2_norm_bwd.Kernels
import VeriTile.Triton.Float.StructuralSafety
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.L2NormBwd.Rewrite
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Local statement agreement includes both execution and the actual-source
safety check. It is used only for the three signed integer pointer products. -/
private def SameStep {A : Type} [Inhabited A] (M : Algebra A)
    (left right : ComputeStmt) : Prop :=
  (∀ s, step M left s = step M right s) ∧
  (∀ bounds s, Safety.stepSafe M bounds left s ↔ Safety.stepSafe M bounds right s)

private theorem same_refl {A : Type} [Inhabited A] (M : Algebra A) (st : ComputeStmt) :
    SameStep M st st := ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩

/-- All fourteen statements are paired in their original order. The row cast
is shared, the next three statements use integer multiplication commutativity,
and the entire floating/memory suffix is identical in both quotations. -/
private theorem source_steps {A : Type} [Inhabited A] (M : Algebra A)
    (layout : Strides) (N B : Nat) :
    List.Forall₂ (SameStep M) (source layout N B).surfaceBody
      (indexCommuted layout N B).surfaceBody := by
  refine .cons (same_refl M _) (.cons ?_ (.cons ?_ (.cons ?_ ?_)))
  · constructor
    · intro s
      cases rowValue : s.regs .int [] "row" <;>
      simp [rowValue,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,
        Option.bind,FP.Structural.numeric,FP.Structural.bop,
        Broadcast.leftIndex,Broadcast.rightIndex,Int.mul_comm]
    · intro bounds s
      simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeOp_unfold]
  · constructor
    · intro s
      cases rowValue : s.regs .int [] "row" <;>
      simp [rowValue,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,
        Option.bind,FP.Structural.numeric,FP.Structural.bop,
        Broadcast.leftIndex,Broadcast.rightIndex,Int.mul_comm]
    · intro bounds s
      simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeOp_unfold]
  · constructor
    · intro s
      cases rowValue : s.regs .int [] "row" <;>
      simp [rowValue,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,
        Option.bind,FP.Structural.numeric,FP.Structural.bop,
        Broadcast.leftIndex,Broadcast.rightIndex,Int.mul_comm]
    · intro bounds s
      simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeOp_unfold]
  · exact List.forall₂_same.mpr (fun st _ => same_refl M st)

private theorem run_eq {A : Type} [Inhabited A] (M : Algebra A)
    {left right : List ComputeStmt} (paired : List.Forall₂ (SameStep M) left right)
    (s : State A) : run M left s = run M right s := by
  induction paired generalizing s with
  | nil => rfl
  | @cons a b left right same rest ih =>
    simp only [run,same.1 s]
    cases step M b s with
    | none => rfl
    | some t => exact ih t

private theorem traceSafe_iff {A : Type} [Inhabited A] (M : Algebra A)
    {left right : List ComputeStmt} (paired : List.Forall₂ (SameStep M) left right)
    (bounds : RegionName → Nat) (s : State A) :
    Safety.traceSafeList M bounds left s ↔ Safety.traceSafeList M bounds right s := by
  induction paired generalizing s with
  | nil => exact Iff.rfl
  | @cons a b left right same rest ih =>
    rw [Safety.traceSafeList_unfold M bounds (a :: left) s,
      Safety.traceSafeList_unfold M bounds (b :: right) s]
    dsimp only
    rw [same.1 s,same.2 bounds s]
    cases step M b s with
    | none => exact Iff.rfl
    | some t => exact and_congr_right (fun _ => ih t)

/-- Complete typed state and failure agreement for the actual repaired source
and candidate, from every state and every floating primitive interpretation.
Only signed integer multiplication is rewritten; no FP law is used. -/
theorem execution_eq {A : Type} [Inhabited A] (M : Algebra A)
    (layout : Strides) (N B : Nat) (s : State A) :
    FP.Structural.exec M (source layout N B) s =
      FP.Structural.exec M (indexCommuted layout N B) s :=
  run_eq M (source_steps M layout N B) s

/-- Safety transport follows identical successor states, including every load,
reduction and store in the common suffix. -/
theorem source_safe_iff {A : Type} [Inhabited A] (M : Algebra A)
    (layout : Strides) (N B : Nat) (bounds : RegionName → Nat) (s : State A) :
    TraceSafe M bounds (source layout N B) s ↔
      TraceSafe M bounds (indexCommuted layout N B) s :=
  traceSafe_iff M (source_steps M layout N B) bounds s

#axiomsClean execution_eq
#axiomsClean source_safe_iff
end
end VeriTile.Bench.Repairs.L2NormBwd.Rewrite
