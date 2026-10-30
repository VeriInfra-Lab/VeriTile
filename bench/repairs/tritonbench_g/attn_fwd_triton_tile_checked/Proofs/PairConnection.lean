import bench.repairs.tritonbench_g.attn_fwd_triton_tile_checked.PairIO
import bench.repairs.tritonbench_g.attn_fwd_triton_tile_checked.Proofs.NativeSchedule
import bench.repairs.tritonbench_g.attn_fwd_triton_tile_checked.Kernels
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.PairConnection
open VeriTile Triton FP.Structural AttnFwdTriton.TypedExecution
noncomputable section
set_option maxHeartbeats 800000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
set_option linter.unusedTactic false
set_option linter.unreachableTactic false
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


theorem remainder (a b : Nat) : a % b=a-(a/b)*b := Nat.mod_eq_sub_div_mul

theorem no_underflow (a b : Nat) : (a/b)*b≤a := Nat.div_mul_le_self a b

theorem step_remainder (M : Algebra A) (out dividend : RegName) (width : Nat) :
    StepEq M
      (.assign .nat [] out (.alg (.mod .nat .nil (.ref .nat [] dividend) (.constNat width))))
      (.assign .nat [] out (.alg (.sub .nat .nil (.ref .nat [] dividend)
        (.mul .nat .nil (.tritonDiv .nat .nil (.ref .nat [] dividend) (.constNat width))
          (.constNat width))))) := by
  constructor
  · intro s
    simp only [step,evalExpr,evalOp_unfold]
    cases ha : s.regs .nat [] dividend
    all_goals simp [ha,bind,Option.bind,numeric,integralMod,integralFloorDiv,bop,←remainder]
    all_goals unfold bop
    all_goals simp only [Broadcast.leftIndex,Broadcast.rightIndex,←remainder]
  · intro bounds s
    simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeOp_unfold]

/-- Full source execution and actual access trace, for the same arbitrary
numeric backend and complete original Config. -/
theorem source_pair (M : Algebra A) (p : Config) (BD : Nat) :
    CodeEq M (source p BD Bool.false).surfaceBody (kernel p BD).surfaceBody := by
  unfold source kernel AttnFwdTriton.TypedSource.repaired quotientRemainder ComputeKernel.surfaceBody
  simp only [Bool.false_eq_true, ↓reduceIte]
  repeat' first
    | exact code_refl M _
    | exact step_refl M _
    | apply code_cons
    | exact step_remainder M _ _ _

#axiomsClean source_pair
end
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.PairConnection
