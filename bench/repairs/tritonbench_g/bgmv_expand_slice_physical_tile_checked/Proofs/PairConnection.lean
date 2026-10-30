import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Kernels
import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Proofs.IOImplementation
import VeriTile.Triton.Float.Loop
import VeriTile.Triton.Float.GuardedRewrite
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
import VeriTile.Triton.Float.ScalarOpsLaws

namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.PairConnection
open VeriTile Triton FP.Structural Direct Direct.FPValues
noncomputable section
set_option maxHeartbeats 250000
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

theorem range_safe {M : Algebra A} {l r : List ComputeStmt} (body : CodeEq M l r)
    (bounds : RegionName → Nat) (idx : RegName) (stop : Int) (stride : Nat)
    (cur : Int) (s : State A) :
    Safety.rangeIntSafe M bounds idx cur stop stride l s ↔
      Safety.rangeIntSafe M bounds idx cur stop stride r s := by
  induction cur using (measure (fun i : Int => (stop-i).toNat)).wf.induction generalizing s with
  | h cur ih =>
    rw [Safety.rangeIntSafe_unfold,Safety.rangeIntSafe_unfold]
    by_cases hz : stride=0
    · simp [hz]
    simp only [hz,↓reduceIte]
    by_cases hc : cur<stop
    · simp only [hc,↓reduceIte,body.safe,body.run]
      cases run M r (s.setReg idx .int [] (fun _ => cur)) with
      | none => rfl
      | some t =>
        exact and_congr_right (fun _ => ih (cur+stride) (by simp_wf;omega) t)
    · simp [hc]

theorem step_range {M : Algebra A} (idx : RegName) (start stop : Op .int []) (stride : Op .nat [])
    {l r : List ComputeStmt} (body : CodeEq M l r) :
    StepEq M (.forRangeInt idx start stop stride l) (.forRangeInt idx start stop stride r) := by
  constructor
  · intro s
    simp only [step]
    cases evalOp M none start s <;> cases evalOp M none stop s <;> cases evalOp M none stride s
    all_goals simp only [bind,Option.bind_none,Option.bind_some]
    rename_i a b d
    exact rangeInt_congr_on_reachable M idx _ _ _ l r s (fun _ t _ _ _ => body.run _)
  · intro bounds s
    simp only [Safety.stepSafe_unfold]
    cases evalOp M none start s <;> cases evalOp M none stop s <;> cases evalOp M none stride s
    all_goals try rfl
    exact and_congr_right (fun _ => and_congr_right (fun _ => and_congr_right
      (fun _ => range_safe body bounds idx _ _ _ s)))

/-- The sole numerical fact used by the source transformation. There is no
finite-value guard: every value in the selected FP32 model is covered. -/
def AdditionCommutes (M : Algebra A) : Prop :=
  ∀ a b, M.binary (some .fp32) .real .add a b = M.binary (some .fp32) .real .add b a

theorem eval_add (M : Algebra A) (comm : AdditionCommutes M)
    (a b : Op .real [n]) (s : State A) :
    evalOp M (some .fp32) (.add .real (.consSame .nil) a b) s =
      evalOp M (some .fp32) (.add .real (.consSame .nil) b a) s := by
  have ea (x y : Op .real [n]) :
      evalOp M (some .fp32) (.add .real (.consSame .nil) x y) s =
      (evalOp M (some .fp32) x s >>= fun va => evalOp M (some .fp32) y s >>= fun vb =>
        some (bop (M.binary (some .fp32) .real .add) (.consSame .nil) va vb)) := rfl
  rw [ea,ea]
  cases evalOp M (some .fp32) a s <;> cases evalOp M (some .fp32) b s
  all_goals simp only [bind,Option.bind_none,Option.bind_some]
  rename_i va vb
  congr 1
  funext i
  exact comm _ _

theorem step_add (M : Algebra A) (comm : AdditionCommutes M)
    (out : RegName) (a b : Op .real [n]) :
    StepEq M
      (.assign .real [n] out (.compute (.alg .fp32
        (.castFloat .real .real (.add .real (.consSame .nil) a b)))))
      (.assign .real [n] out (.compute (.alg .fp32
        (.castFloat .real .real (.add .real (.consSame .nil) b a))))) := by
  constructor
  · intro s
    simp only [step,FP.Structural.evalExpr,evalComputeOp]
    have ec (x : Op .real [n]) : evalOp M (some .fp32) (.castFloat .real .real x) s =
        (evalOp M (some .fp32) x s >>= fun v => some (fun i => M.cast (some .fp32) .real .real (v i))) := rfl
    erw [ec,ec,eval_add M comm a b s]
    rfl
  · intro bounds s
    simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold]
    have sa (x y : Op .real [n]) :
        Safety.safeOp M (some .fp32) bounds s
          (.castFloat .real .real (.add .real (.consSame .nil) x y)) =
        (Safety.safeOp M (some .fp32) bounds s x ∧ Safety.safeOp M (some .fp32) bounds s y) := by
      rw [Safety.safeOp_unfold]
      dsimp only [FloatDType.toTileDType]
      rw [Safety.safeOp_unfold]
    exact (Iff.of_eq (sa a b)).trans (and_comm.trans (Iff.of_eq (sa b a)).symm)

/-- Full quoted-source connection. No index, mask, loop or memory expression
is abstracted into a numerical premise. -/
theorem source_pair (M : Algebra A) (comm : AdditionCommutes M) (h : Call) :
    CodeEq M (Kernels.source h).surfaceBody (Kernels.outputFirst h).surfaceBody := by
  unfold Kernels.source Surface.source Kernels.outputFirst ComputeKernel.surfaceBody
  repeat' first
    | exact code_refl M _
    | exact step_refl M _
    | apply code_cons
    | apply step_if
    | apply step_if_else
    | apply step_range
    | exact step_add M comm _ _ _

#axiomsClean source_pair
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.PairConnection
