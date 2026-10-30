import bench.repairs.tritonbench_g.chunk_delta_fwd_static_dot.Kernels
import VeriTile.Triton.Float.StructuralSafety
import VeriTile.Triton.Float.Loop

/-! Exact source connection for specialization-time address product folding.
Only natural-number address arithmetic changes. Every floating primitive and
compiler staging node is preserved, for arbitrary numerical interpretations. -/
namespace VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.PairSource
open VeriTile Triton FP.Structural FP.Structural.Safety
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

structure CodeAgree (M : Algebra α) (lhs rhs : List ComputeStmt) : Prop where
  runs : ∀ s, run M lhs s = run M rhs s
  safe : ∀ bounds s, traceSafeList M bounds lhs s ↔ traceSafeList M bounds rhs s
structure StmtAgree (M : Algebra α) (lhs rhs : ComputeStmt) : Prop where
  runs : ∀ s, FP.Structural.step M lhs s = FP.Structural.step M rhs s
  safe : ∀ bounds s, stepSafe M bounds lhs s ↔ stepSafe M bounds rhs s

theorem code_refl (M : Algebra α) (code : List ComputeStmt) : CodeAgree M code code :=
  ⟨fun _ => rfl, fun _ _ => Iff.rfl⟩
theorem stmt_refl (M : Algebra α) (st : ComputeStmt) : StmtAgree M st st :=
  ⟨fun _ => rfl, fun _ _ => Iff.rfl⟩

theorem cons (M : Algebra α) {a b : ComputeStmt} {xs ys : List ComputeStmt}
    (head : StmtAgree M a b) (tail : CodeAgree M xs ys) :
    CodeAgree M (a::xs) (b::ys) := by
  constructor
  · intro s
    simp only [run, head.runs s]
    exact congrArg (fun f => (FP.Structural.step M b s).bind f) (funext tail.runs)
  · intro bounds s
    simp only [traceSafeList_cons, head.runs s, head.safe bounds s]
    constructor <;> rintro ⟨h, rest⟩ <;> refine ⟨h, ?_⟩
    · intro t ht; exact (tail.safe bounds t).mp (rest t ht)
    · intro t ht; exact (tail.safe bounds t).mpr (rest t ht)

theorem branch (M : Algebra α) (cond : Op .bool []) {xs ys : List ComputeStmt}
    (body : CodeAgree M xs ys) : StmtAgree M (.ifThen cond xs) (.ifThen cond ys) := by
  constructor
  · intro s
    simp only [FP.Structural.step]
    cases he : evalOp M none cond s with
    | none => rfl
    | some c => simp only [bind, Option.bind_some]; split <;> simp [body.runs]
  · intro bounds s
    simp only [stepSafe_unfold]
    cases he : evalOp M none cond s with
    | none => simp
    | some c => simp only; split <;> simp [body.safe]

theorem range_stmt (M : Algebra α) (idx : RegName) (start stop stride : Nat)
    {xs ys : List ComputeStmt} (body : CodeAgree M xs ys) :
    StmtAgree M (.forRange idx start stop stride xs) (.forRange idx start stop stride ys) := by
  constructor
  · intro s
    simp only [FP.Structural.step]
    apply range_congr_on_reachable
    intro cur t _ _ _
    exact body.runs _
  · intro bounds s
    simp only [stepSafe_unfold]
    have go : ∀ cur s, rangeSafe M bounds idx cur stop stride xs s ↔
        rangeSafe M bounds idx cur stop stride ys s := by
      intro cur
      induction cur using (measure (fun i => stop-i)).wf.induction with
      | h cur ih =>
        intro s
        rw [rangeSafe_unfold, rangeSafe_unfold]
        by_cases hz : stride=0
        · simp [hz]
        simp only [hz, if_false]
        by_cases hc : cur<stop
        · simp only [hc, if_true, body.runs, body.safe]
          cases he : run M ys (s.setReg idx .nat [] (fun _ => cur)) with
          | none => simp
          | some t => simp only; rw [ih (cur+stride) (by simp_wf; omega) t]
        · simp [hc]
    exact go start s


end
end VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.PairSource
