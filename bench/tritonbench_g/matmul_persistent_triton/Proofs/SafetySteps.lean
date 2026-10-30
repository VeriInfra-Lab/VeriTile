import bench.tritonbench_g.matmul_persistent_triton.Model
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.SourceSafety
open VeriTile.Triton FP.Structural Core
open VeriTile.Bench.Repairs.MatmulPersistentTriton
noncomputable section
set_option maxHeartbeats 600000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem safe_list (M : Algebra α) (bounds : RegionName → Nat)
    (code : List ComputeStmt) (safe : ∀ st∈code,∀ s,Safety.stepSafe M bounds st s)
    (s : State α) : Safety.traceSafeList M bounds code s := by
  induction code generalizing s with
  | nil => exact Safety.traceSafeList_nil _ _ _
  | cons st rest ih =>
    apply Safety.traceSafeList_cons_intro (safe st (by simp) s)
    intro t _
    exact ih (fun st h => safe st (by simp [h])) t

theorem prelude_safe (P : Source.Arithmetic α α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (backend P) bounds (prelude p) s := by
  apply safe_list
  intro st hs t
  rw [prelude_raw] at hs
  simp only [rawPrelude,TypedSource.kernel,TypedSource.fp16,TypedSource.f8e4,ComputeKernel.surfaceBody,List.take,List.mem_cons,List.not_mem_nil,or_false] at hs
  rcases hs with h|h|h|h|h|h|h|h|h|h|h|h|h|h|h
  all_goals subst st
  all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,Safety.traceSafeList_unfold,step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind]
  all_goals split <;> try trivial
  all_goals intro _; split <;> trivial

theorem tile_prelude_safe (P : Source.Arithmetic α α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (backend P) bounds (tilePrelude p) s := by
  apply safe_list
  intro st hs t
  rw [tilePrelude_raw] at hs
  simp [rawTilePrelude,TypedSource.kernel,TypedSource.fp16,TypedSource.f8e4,ComputeKernel.surfaceBody,List.getElem?_eq_getElem,
    List.take,List.mem_cons,List.not_mem_nil,or_false] at hs
  rcases hs with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl
  all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]


#axiomsClean prelude_safe
#axiomsClean tile_prelude_safe
end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.SourceSafety
