import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.NativeObservation
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.PairConnection
open VeriTile Triton FP.Structural Whole
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 40000
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
    simp only [FP.Structural.run,head.step]
    cases FP.Structural.step M r s with
    | none => rfl
    | some t => exact tail.run t
  · intro bounds s
    simp only [Safety.traceSafeList_cons,head.safe,head.step]
    exact and_congr_right (fun _ => forall_congr' (fun t => imp_congr_right (fun _ => tail.safe bounds t)))

theorem bound_identity (width length start : Nat) :
    (if width<length-start then width else length-start)=
      (if start+width<length then start+width else length)-start := by
  split_ifs <;> omega

/-- Proof decomposition of the exact integer source expressions. -/
def startOp : Op .nat [] :=
  .mul .nat .nil (.ref .nat [] "i_t") (.constNat 64)
def minOp (a b : Op .nat []) : Op .nat [] := .where (.lt .nat .nil a b) a b
def sourceBound (length : Nat) : Op .nat [] :=
  minOp (.constNat 64) (.sub .nat .nil (.constNat length) startOp)
def candidateBound (length : Nat) : Op .nat [] :=
  .sub .nat .nil (minOp (.add .nat .nil startOp (.constNat 64)) (.constNat length)) startOp

theorem eval_bound (M : Algebra A) (p : Option ComputeDType) (length : Nat) (s : State A) :
    evalOp M p (sourceBound length) s=evalOp M p (candidateBound length) s := by
  simp only [sourceBound,candidateBound,minOp,startOp,evalOp_unfold]
  cases ht : s.regs .nat [] "i_t"
  all_goals simp [ht,bind,Option.bind,numeric,numericLt,natLt,bop,
    Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals unfold bop
  all_goals funext i
  all_goals exact bound_identity _ _ _

def gateStmt (bound : Op .nat []) : ComputeStmt :=
  .assign .real [64] "b_dg" (.compute (.alg .fp32
    (.where (.lt .nat .scalarR (.ref .nat [64] "o_i")
      (.sub .nat .nil bound (.constNat 1)))
      (.ref .real [64] "b_dg")
      (.add .real (.consR .nil) (.ref .real [64] "b_dg") (.ref .real [1] "b_dg_last")))))

theorem eval_minus (M : Algebra A) (p : Option ComputeDType) (length : Nat) (s : State A) :
    evalOp M p (.sub .nat .nil (sourceBound length) (.constNat 1)) s=
      evalOp M p (.sub .nat .nil (candidateBound length) (.constNat 1)) s := by
  change (do
      let va ← evalOp M p (sourceBound length) s
      let vb ← evalOp M p (.constNat 1) s
      pure (bop (numeric M p .sub .nat) .nil va vb)) =
    (do
      let va ← evalOp M p (candidateBound length) s
      let vb ← evalOp M p (.constNat 1) s
      pure (bop (numeric M p .sub .nat) .nil va vb))
  rw [eval_bound]

theorem eval_condition (M : Algebra A) (p : Option ComputeDType) (length : Nat) (s : State A) :
    evalOp M p (.lt .nat .scalarR (.ref .nat [64] "o_i")
      (.sub .nat .nil (sourceBound length) (.constNat 1))) s=
    evalOp M p (.lt .nat .scalarR (.ref .nat [64] "o_i")
      (.sub .nat .nil (candidateBound length) (.constNat 1))) s := by
  change (do
      let f ← numericLt M p .nat
      let va ← evalOp M p (.ref .nat [64] "o_i") s
      let vb ← evalOp M p (.sub .nat .nil (sourceBound length) (.constNat 1)) s
      pure (bop f .scalarR va vb)) =
    (do
      let f ← numericLt M p .nat
      let va ← evalOp M p (.ref .nat [64] "o_i") s
      let vb ← evalOp M p (.sub .nat .nil (candidateBound length) (.constNat 1)) s
      pure (bop f .scalarR va vb))
  rw [eval_minus]

-- Covers every register state, including missing values and failing execution.
theorem gate_step (M : Algebra A) (length : Nat) :
    StepEq M (gateStmt (sourceBound length)) (gateStmt (candidateBound length)) := by
  have values (s : State A) :
      evalOp M (some .fp32) (.where
        (.lt .nat .scalarR (.ref .nat [64] "o_i")
          (.sub .nat .nil (sourceBound length) (.constNat 1)))
        (.ref .real [64] "b_dg")
        (.add .real (.consR .nil) (.ref .real [64] "b_dg") (.ref .real [1] "b_dg_last"))) s=
      evalOp M (some .fp32) (.where
        (.lt .nat .scalarR (.ref .nat [64] "o_i")
          (.sub .nat .nil (candidateBound length) (.constNat 1)))
        (.ref .real [64] "b_dg")
        (.add .real (.consR .nil) (.ref .real [64] "b_dg") (.ref .real [1] "b_dg_last"))) s := by
    change (do
        let vc ← evalOp M (some .fp32) (.lt .nat .scalarR (.ref .nat [64] "o_i")
          (.sub .nat .nil (sourceBound length) (.constNat 1))) s
        let va ← evalOp M (some .fp32) (.ref .real [64] "b_dg") s
        let vb ← evalOp M (some .fp32) (.add .real (.consR .nil)
          (.ref .real [64] "b_dg") (.ref .real [1] "b_dg_last")) s
        pure (fun i => if vc i then va i else vb i)) =
      (do
        let vc ← evalOp M (some .fp32) (.lt .nat .scalarR (.ref .nat [64] "o_i")
          (.sub .nat .nil (candidateBound length) (.constNat 1))) s
        let va ← evalOp M (some .fp32) (.ref .real [64] "b_dg") s
        let vb ← evalOp M (some .fp32) (.add .real (.consR .nil)
          (.ref .real [64] "b_dg") (.ref .real [1] "b_dg_last")) s
        pure (fun i => if vc i then va i else vb i))
    rw [eval_condition]
  constructor
  · intro s
    simp only [gateStmt,FP.Structural.step,evalExpr,evalComputeOp]
    exact congrArg (fun v : Option (Values A .real [64]) =>
      v.bind (fun w => some (s.setReg "b_dg" .real [64] w))) (values s)
  · intro bounds s
    simp [gateStmt,sourceBound,candidateBound,minOp,startOp,
      Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeOp_unfold]

/-- Complete exact source state and actual access-safety transport. -/
theorem source_pair (M : Algebra A) (c : Config) :
    CodeEq M (Whole.kernel c).surfaceBody (kernel c).surfaceBody := by
  have gate := gate_step M c.T
  unfold gateStmt sourceBound candidateBound minOp startOp at gate
  unfold Whole.kernel kernel TypedSource.original TypedSource.specialize
    TypedSource.tailBoundCSE TypedSource.Storage.dtype ComputeKernel.surfaceBody
  repeat' first
    | exact code_refl M _
    | exact step_refl M _
    | apply code_cons
    | exact gate

#axiomsClean bound_identity
#axiomsClean eval_bound
#axiomsClean gate_step
#axiomsClean source_pair
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.PairConnection
