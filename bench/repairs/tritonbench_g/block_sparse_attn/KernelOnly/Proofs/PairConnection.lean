import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.PairIO
import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Proofs.PublicCorrectness
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PairConnection
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1200000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs Bool.false
set_option linter.unusedTactic Bool.false
set_option linter.unreachableTactic Bool.false
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
    simp only [FP.Structural.run,head.step]
    cases FP.Structural.step M r s with
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
    simp only [FP.Structural.step,evalExpr,evalOp_unfold]
    cases ha : s.regs .nat [] dividend
    all_goals simp [ha,bind,Option.bind,numeric,integralMod,integralFloorDiv,bop,←remainder]
    all_goals unfold bop
    all_goals simp only [Broadcast.leftIndex,Broadcast.rightIndex,←remainder]
  · intro bounds s
    simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeOp_unfold]

/-- The complete numerical/CSR suffix is unchanged. This links the ordinary
candidate quotation to exactly the same selected expression-site lowering. -/
theorem source_suffix (p : Parameters) :
    (quotientRemainder p).surfaceBody.drop 27=(source p).surfaceBody.drop 27 := rfl

/-- The candidate uses the existing local native lowering only in the common
unchanged numerical suffix; its head-index prelude comes from its own source. -/


theorem source_pair (M : Algebra A) (p : Parameters) :
    CodeEq M (source p).surfaceBody (quotientRemainder p).surfaceBody := by
  unfold source declared quotientRemainder ComputeKernel.surfaceBody
  repeat' first
    | exact code_refl M _
    | exact step_refl M _
    | apply code_cons
    | exact step_remainder M _ _ _

/-- Same selected live-FMA and seeded-MMA meanings on both complete programs. -/
theorem native_pair (M : Algebra A) (p : Parameters) :
    CodeEq M (selectedSource p) (kernel p).surfaceBody := by
  unfold kernel selectedCandidate selectedSource prelude source declared quotientRemainder ComputeKernel.surfaceBody
  repeat' first
    | exact code_refl M _
    | exact step_refl M _
    | apply code_cons
    | exact step_remainder M _ _ _

#axiomsClean source_suffix
#axiomsClean source_pair
#axiomsClean native_pair
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PairConnection
