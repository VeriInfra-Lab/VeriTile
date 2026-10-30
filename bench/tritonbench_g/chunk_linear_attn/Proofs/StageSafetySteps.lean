import bench.tritonbench_g.chunk_linear_attn.Proofs.StageSafetyBasics

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.StageSafety
open VeriTile Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

/-- Continue from the actual assignment result without expanding the numerical
expression. Every successful result is covered by the tail premise. -/
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

/-- Unfold concrete offset lists before the recursively defined safety check;
this avoids expanding safety under an unknown list member. -/
theorem safeBlockPtr2 (M : Algebra α) (precision : Option ComputeDType)
    (bounds : RegionName → Nat) (s : State α) (region : RegionName)
    (base : Op .nat []) (parent block strides : List Nat) (a b : Op .nat []) :
    Safety.safeOp M precision bounds s (.makeBlockPtrDynOffsets region base parent block strides [a,b]) ↔
      Safety.safeOp M precision bounds s base ∧ Safety.safeOp M precision bounds s a ∧
        Safety.safeOp M precision bounds s b := by
  rw [Safety.safeOp_unfold]
  simp only [List.mem_cons,List.not_mem_nil,or_false,forall_eq_or_imp,forall_eq]

end
end VeriTile.Bench.Repairs.ChunkLinearAttn.StageSafety
