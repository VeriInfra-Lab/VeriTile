import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.TypedGrid
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.TypedSafetyBasics
open VeriTile Triton FP.Structural FP.ProfiledRewrite Whole Typed
open FP.Structural.Safety
noncomputable section
variable {α : Type} [Inhabited α]

/-- Unroll concrete offset lists before recursively checking their expressions. -/
theorem safeBlockPtr1 (M : Algebra α) (p : Option ComputeDType)
    (bounds : RegionName → Nat) (s : State α) (r : RegionName)
    (base : Op .nat []) (parent block strides : List Nat) (a : Op .nat []) :
    safeOp M p bounds s (.makeBlockPtrDynOffsets r base parent block strides [a]) ↔
      safeOp M p bounds s base ∧ safeOp M p bounds s a := by
  rw [safeOp_unfold]
  simp only [List.mem_cons,List.not_mem_nil,or_false,forall_eq]

theorem safeBlockPtr2 (M : Algebra α) (p : Option ComputeDType)
    (bounds : RegionName → Nat) (s : State α) (r : RegionName)
    (base : Op .nat []) (parent block strides : List Nat) (a b : Op .nat []) :
    safeOp M p bounds s (.makeBlockPtrDynOffsets r base parent block strides [a,b]) ↔
      safeOp M p bounds s base ∧ safeOp M p bounds s a ∧ safeOp M p bounds s b := by
  rw [safeOp_unfold]
  simp only [List.mem_cons,List.not_mem_nil,or_false,forall_eq_or_imp,forall_eq]

theorem pure_trace (M : Algebra α) (bounds : RegionName → Nat)
    (code : List ComputeStmt)
    (hs : ∀ st ∈ code, ∀ s : State α,stepSafe M bounds st s)
    (s : State α) : traceSafeList M bounds code s := by
  induction code generalizing s with
  | nil=>exact traceSafeList_nil _ _ _
  | cons st rest ih=>
    apply traceSafeList_cons_intro (hs st (by simp) s)
    intro t _
    exact ih (fun st h=>hs st (by simp [h])) t

end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.TypedSafetyBasics
