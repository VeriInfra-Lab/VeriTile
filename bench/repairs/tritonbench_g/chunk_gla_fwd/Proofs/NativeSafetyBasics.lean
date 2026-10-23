import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeLaunch
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section
variable {α : Type} [Inhabited α]

theorem singleton (M : Algebra α) (bounds : RegionName → Nat)
    (statement : ComputeStmt) (s : State α) :
    Safety.traceSafeList M bounds [statement] s ↔ Safety.stepSafe M bounds statement s := by
  rw [Safety.traceSafeList_cons]
  simp

theorem pure_trace (M : Algebra α) (bounds : RegionName → Nat) (code : List ComputeStmt)
    (safe : ∀ statement ∈ code, ∀ s, Safety.stepSafe M bounds statement s) (s : State α) :
    Safety.traceSafeList M bounds code s := by
  induction code generalizing s with
  | nil => exact Safety.traceSafeList_nil _ _ _
  | cons statement rest ih =>
    apply Safety.traceSafeList_cons_intro (safe statement (by simp) s)
    intro t _
    exact ih (fun stmt mem => safe stmt (by simp [mem])) t

theorem assign_cons (M : Algebra α) (bounds : RegionName → Nat)
    (dtype shape name) (expr : ComputeExpr dtype shape) (rest : List ComputeStmt)
    (s : State α) (safe : Safety.safeExpr M bounds s expr)
    (tail : ∀ value, Safety.traceSafeList M bounds rest (s.setReg name dtype shape value)) :
    Safety.traceSafeList M bounds (.assign dtype shape name expr :: rest) s := by
  apply Safety.traceSafeList_cons_intro
  · simpa only [Safety.stepSafe_unfold] using safe
  · intro t h
    simp only [step] at h
    cases he : evalExpr M expr s with
    | none => simp [he] at h
    | some value =>
      simp only [he,bind,pure,Option.bind_some,Option.some.injEq] at h
      subst t
      exact tail value

theorem safeBlockPtr2 (M : Algebra α) (precision : Option ComputeDType)
    (bounds : RegionName → Nat) (s : State α) (region : RegionName)
    (base : Op .nat []) (parent block strides : List Nat) (a b : Op .nat []) :
    Safety.safeOp M precision bounds s (.makeBlockPtrDynOffsets region base parent block strides [a,b]) ↔
      Safety.safeOp M precision bounds s base ∧ Safety.safeOp M precision bounds s a ∧
        Safety.safeOp M precision bounds s b := by
  rw [Safety.safeOp_unfold]
  simp only [List.mem_cons,List.not_mem_nil,or_false,forall_eq_or_imp,forall_eq]

structure MergeCapacity (p : FPMerge.Layout) (s : State α) (bounds : RegionName → Nat) : Prop where
  input : ∀ f, f < p.NK → ∀ i : FPMerge.Index p, FPMerge.row p s i.1.val < p.T →
    FPMerge.inputAddress p s f i < bounds "scratch"
  output : ∀ i : FPMerge.Index p, FPMerge.active p s i →
    FPMerge.outputAddress p s i < bounds "attention"

end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeSafety
