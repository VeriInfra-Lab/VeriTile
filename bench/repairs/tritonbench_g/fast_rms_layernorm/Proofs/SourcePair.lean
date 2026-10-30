import bench.repairs.tritonbench_g.fast_rms_layernorm.Kernels
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.SourcePair
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def original (h : Call) (backward : Bool) : ComputeKernel :=
  if backward then Kernels.backward h else Kernels.forwardRuntime h

def source (candidate : Bool) (h : Call) (backward : Bool) : ComputeKernel :=
  if backward then if candidate then Kernels.backwardIndexCommuted h else Kernels.backward h
  else if candidate then Kernels.forwardRuntimeIndexCommuted h else Kernels.forwardRuntime h

private def SameStep {α : Type} [Inhabited α] (M : Algebra α)
    (left right : ComputeStmt) : Prop :=
  (∀ s, step M left s = step M right s) ∧
  (∀ bounds s, Safety.stepSafe M bounds left s ↔ Safety.stepSafe M bounds right s)

private theorem same_refl {α : Type} [Inhabited α] (M : Algebra α) (st : ComputeStmt) :
    SameStep M st st := ⟨fun _ => rfl, fun _ _ => Iff.rfl⟩

/-- Pair the complete actual bodies. Only the fifth statement changes, and
its transformation is signed integer multiplication commutativity. -/
private theorem source_steps {α : Type} [Inhabited α] (M : Algebra α)
    (h : Call) (backward : Bool) :
    List.Forall₂ (SameStep M) (source Bool.false h backward).surfaceBody
      (source Bool.true h backward).surfaceBody := by
  cases backward <;> cases mode : h.gemma
  all_goals
    simp only [source, Kernels.forwardRuntime, Kernels.forwardRuntimeIndexCommuted,
      Kernels.backward, Kernels.backwardIndexCommuted, mode, Bool.false_eq_true, ↓reduceIte]
    refine .cons (same_refl M _) (.cons (same_refl M _)
      (.cons (same_refl M _) (.cons (same_refl M _) (.cons ?_ ?_))))
    · constructor
      · intro s
        cases rowValue : s.regs .int [] "row_idx" <;>
          simp [rowValue, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas,
            Option.bind, numeric, bop, Broadcast.leftIndex, Broadcast.rightIndex, Int.mul_comm]
      · intro bounds s
        simp [Safety.stepSafe_unfold, Safety.safeExpr_unfold, Safety.safeOp_unfold]
    · exact List.forall₂_same.mpr (fun st _ => same_refl M st)

private theorem paired_run {α : Type} [Inhabited α] (M : Algebra α)
    {left right : List ComputeStmt} (paired : List.Forall₂ (SameStep M) left right)
    (s : State α) : run M left s = run M right s := by
  induction paired generalizing s with
  | nil => rfl
  | @cons a b left right same rest ih =>
    simp only [run, same.1 s]
    cases step M b s with
    | none => rfl
    | some t => exact ih t

private theorem paired_safe {α : Type} [Inhabited α] (M : Algebra α)
    {left right : List ComputeStmt} (paired : List.Forall₂ (SameStep M) left right)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList M bounds left s ↔ Safety.traceSafeList M bounds right s := by
  induction paired generalizing s with
  | nil => exact Iff.rfl
  | @cons a b left right same rest ih =>
    rw [Safety.traceSafeList_unfold M bounds (a :: left) s,
      Safety.traceSafeList_unfold M bounds (b :: right) s]
    dsimp only
    rw [same.1 s, same.2 bounds s]
    cases step M b s with
    | none => exact Iff.rfl
    | some t => exact and_congr_right (fun _ => ih t)

/-- The complete state, including failure, agrees for every floating model.
Successful full-grid IO is proved separately from the actual source. -/
theorem source_exec {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (h : Call) (backward : Bool) :
    exec M (source Bool.false h backward) s = exec M (source Bool.true h backward) s :=
  paired_run M (source_steps M h backward) s

theorem source_safe {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (bounds : RegionName → Nat) (h : Call) (backward : Bool) :
    TraceSafe M bounds (source Bool.false h backward) s ↔
      TraceSafe M bounds (source Bool.true h backward) s :=
  paired_safe M (source_steps M h backward) bounds s

#axiomsClean source_exec
#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.SourcePair
