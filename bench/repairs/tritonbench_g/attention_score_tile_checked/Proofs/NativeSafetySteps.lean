import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.FPOutput
import bench.repairs.tritonbench_g.attention_score_tile_checked.Kernels
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.NativeSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite Safety Reused FPEngine FPContext
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Resolve fixed offset lists before recursively unfolding safety. -/
theorem safeBlockPtr2 (M : Algebra α) (precision : Option ComputeDType)
    (bounds : RegionName → Nat) (s : State α) (region : RegionName)
    (base : Op .nat []) (parent block strides : List Nat) (a b : Op .nat []) :
    safeOp M precision bounds s (.makeBlockPtrDynOffsets region base parent block strides [a,b]) ↔
      safeOp M precision bounds s base ∧ safeOp M precision bounds s a ∧ safeOp M precision bounds s b := by
  rw [safeOp_unfold]
  simp only [List.mem_cons,List.not_mem_nil,or_false,forall_eq_or_imp,forall_eq]

theorem safeAdvance2 (M : Algebra α) (precision : Option ComputeDType)
    (bounds : RegionName → Nat) (s : State α) (shape : TileShape)
    (ptr : Op .blockPtr shape) (a b : Op .int []) :
    safeOp M precision bounds s (.advanceBlockPtrDyn ptr [a,b]) ↔
      safeOp M precision bounds s ptr ∧ safeOp M precision bounds s a ∧ safeOp M precision bounds s b := by
  rw [safeOp_unfold]
  simp only [List.mem_cons,List.not_mem_nil,or_false,forall_eq_or_imp,forall_eq]

theorem pure_safe (M : Algebra α) (bounds : RegionName → Nat) (code : List ComputeStmt)
    (pure : ∀ st∈code,∀ s : State α,stepSafe M bounds st s) (s : State α) :
    traceSafeList M bounds code s := by
  induction code generalizing s with
  | nil => exact traceSafeList_nil _ _ _
  | cons st rest ih =>
    apply traceSafeList_cons_intro (pure st (by simp) s)
    intro next _
    exact ih (fun stmt member => pure stmt (by simp [member])) next

theorem prelude_tail (M : Algebra α) (dot : Dot α) (B D : Nat) (p : RealSource.Parameters)
    (s : State α) (bounds : RegionName → Nat) :
    traceSafeList (engine (withDot M dot)) bounds ((parts B D p).before.drop 12) s := by
  apply pure_safe
  intro stmt member current
  by_cases hn : p.evenN=Bool.true
  all_goals simp [parts,FPEngine.kernel,RealSource.source,TypedSource.original,Whole002.TypedSource.widened,hn,
    ComputeKernel.surfaceBody,Source.intRange?,Source.prefixBefore,
    Option.map,Option.getD,List.drop_succ_cons,List.drop_zero,List.drop_nil,
    List.mem_cons,List.not_mem_nil,or_false,ite_true,ite_false] at member
  all_goals rcases member with rfl|rfl|rfl|rfl
  all_goals simp only [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold]
  all_goals try rw [safeBlockPtr2]
  all_goals try rw [safeAdvance2]
  all_goals simp only [safeOp_unfold,true_and,and_true]

theorem body_tail (M : Algebra α) (dot : Dot α) (B D : Nat) (p : RealSource.Parameters)
    (s : State α) (bounds : RegionName → Nat) :
    traceSafeList (engine (withDot M dot)) bounds ((parts B D p).body.drop 3) s := by
  apply pure_safe
  intro stmt member current
  by_cases hm : p.evenM=Bool.true
  all_goals by_cases hw : p.window=Bool.true
  all_goals by_cases hc : p.complement=Bool.true
  all_goals simp [parts,FPEngine.kernel,RealSource.source,TypedSource.original,Whole002.TypedSource.widened,hm,hw,hc,
    ComputeKernel.surfaceBody,Source.intRange?,Source.prefixBefore,
    Option.map,Option.getD,List.drop_succ_cons,List.drop_zero,List.drop_nil,
    List.mem_cons,List.not_mem_nil,or_false,ite_true,ite_false] at member
  all_goals rcases member with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl
  all_goals simp only [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold]
  all_goals try rw [safeBlockPtr2]
  all_goals try rw [safeAdvance2]
  all_goals simp [safeAdvance2,safeBlockPtr2,safeOp_unfold,true_and,and_true,traceSafeList_unfold,stepSafe_unfold,
    safeExpr_unfold,safeCompute_unfold,evalOp_unfold,hm,hw,hc]
  all_goals repeat' first | exact True.intro | constructor | split

end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.NativeSafety
