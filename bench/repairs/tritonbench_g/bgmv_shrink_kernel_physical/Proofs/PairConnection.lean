import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Kernels
import VeriTile.Triton.Float.GuardedRewrite
import VeriTile.Triton.Float.StructuralSafety
import VeriTile.Triton.Float.ScalarOpsLaws

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.PairConnection
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 400000
set_option maxRecDepth 24000
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

/-- The only numerical rewrite exchanges the final FP32 multiplication.
It covers every value of the selected model, without a finite-value guard. -/
def MultiplicationCommutes (M : Algebra A) : Prop :=
  ∀ a b, M.binary (some .fp32) .real .mul a b = M.binary (some .fp32) .real .mul b a

theorem eval_mul (M : Algebra A) (comm : MultiplicationCommutes M)
    (a : Op .real [n]) (b : Op .real []) (s : State A) :
    evalOp M (some .fp32) (.mul .real .scalarR a b) s =
      evalOp M (some .fp32) (.mul .real .scalarL b a) s := by
  have er : evalOp M (some .fp32) (.mul .real .scalarR a b) s =
      (evalOp M (some .fp32) a s >>= fun va => evalOp M (some .fp32) b s >>= fun vb =>
        some (bop (M.binary (some .fp32) .real .mul) .scalarR va vb)) := rfl
  have el : evalOp M (some .fp32) (.mul .real .scalarL b a) s =
      (evalOp M (some .fp32) b s >>= fun vb => evalOp M (some .fp32) a s >>= fun va =>
        some (bop (M.binary (some .fp32) .real .mul) .scalarL vb va)) := rfl
  rw [er,el]
  cases evalOp M (some .fp32) a s <;> cases evalOp M (some .fp32) b s
  all_goals simp only [bind,Option.bind_none,Option.bind_some]
  rename_i va vb
  congr 1
  funext i
  exact comm _ _

theorem step_mul (M : Algebra A) (comm : MultiplicationCommutes M)
    (out : RegName) (a : Op .real [n]) (b : Op .real []) :
    StepEq M
      (.assign .real [n] out (.compute (.alg .fp32 (.mul .real .scalarR a b))))
      (.assign .real [n] out (.compute (.alg .fp32 (.mul .real .scalarL b a)))) := by
  constructor
  · intro s
    simp only [step,FP.Structural.evalExpr,evalComputeOp]
    erw [eval_mul M comm a b s]
    rfl
  · intro bounds s
    simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold]
    rw [Safety.safeOp_unfold,Safety.safeOp_unfold]
    exact and_comm

/-- Equality of the complete ordinary source quotations, preserving failures,
registers, memory, access safety, masks, signed K loop and atomic operations. -/
theorem source_pair (M : Algebra A) (comm : MultiplicationCommutes M) (h : Call) :
    CodeEq M (Kernels.source h).surfaceBody (Kernels.scalarLeft h).surfaceBody := by
  unfold Kernels.source Kernels.scalarLeft
  cases h.storage <;>
    unfold Kernels.originalFP16 Kernels.scalarLeftFP16 Kernels.originalBF16 Kernels.scalarLeftBF16
      ComputeKernel.surfaceBody
  all_goals
    repeat' first
      | exact code_refl M _
      | exact step_refl M _
      | apply code_cons
      | apply step_if
      | exact step_mul M comm _ _ _

#axiomsClean source_pair
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.PairConnection
