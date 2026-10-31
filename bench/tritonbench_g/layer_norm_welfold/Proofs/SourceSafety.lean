import bench.tritonbench_g.layer_norm_welfold.Proofs.SourceBinding
import bench.tritonbench_g.layer_norm_welfold.Proofs.Row
import bench.tritonbench_g.fused_layernorm_triton.Proofs.SourceSafety

namespace VeriTile.Bench.Repairs.LayerNormWelfold.SourceSafety
open VeriTile.Triton FP.Structural Reuse
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem append_safe (M : Algebra α) (bounds : RegionName → Nat)
    (a b : List ComputeStmt) (s t : State α)
    (ha : Safety.traceSafeList M bounds a s) (ran : run M a s = some t)
    (hb : Safety.traceSafeList M bounds b t) :
    Safety.traceSafeList M bounds (a++b) s := by
  apply (Safety.traceSafeList_append M bounds a b s).mpr
  refine ⟨ha,?_⟩
  intro u hu
  have eq := Option.some.inj (ran.symm.trans hu)
  simpa only [←eq] using hb

private theorem list_safe (M : Algebra α) (bounds : RegionName → Nat)
    (code : List ComputeStmt)
    (each : ∀ st ∈ code, ∀ s : State α, Safety.stepSafe M bounds st s) (s : State α) :
    Safety.traceSafeList M bounds code s := by
  induction code generalizing s with
  | nil => exact Safety.traceSafeList_nil M bounds s
  | cons head tail ih =>
    apply Safety.traceSafeList_cons_intro (each head List.mem_cons_self s)
    intro t _
    exact ih (fun st h => each st (List.mem_cons_of_mem _ h)) t

theorem init_safe (M : Algebra α) (s : State α) (S B : Nat) (bounds : RegionName → Nat) :
    Safety.traceSafeList M bounds (init S B) s := by
  apply list_safe
  intro st member t
  simp only [init,ComputeKernel.surfaceBody,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl
  all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]

theorem reset_safe (M : Algebra α) (s : State α) (B : Nat) (bounds : RegionName → Nat) :
    Safety.traceSafeList M bounds (reset B) s := by
  apply list_safe
  intro st member t
  simp only [reset,ComputeKernel.surfaceBody,List.mem_singleton] at member
  subst st
  simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]

theorem chunk_safe (M : Algebra α) (phase : Bool) (s t : State α)
    (X : RegionName) (D B k : Nat) (values : Fin B → α) (mean : α)
    (h : Ready s t values mean) (bounds : RegionName → Nat)
    (hx : ∀ j, j < D → j+D*s.pids 0 < bounds X) :
    Safety.traceSafeList M bounds (chunk phase X D B k) t := by
  have split : chunk phase X D B k =
      (chunk phase X D B k).take 4 ++ (chunk phase X D B k).drop 4 :=
    (List.take_append_drop 4 _).symm
  rw [split,Safety.traceSafeList_append]
  constructor
  · have last (stmt : ComputeStmt) (state : State α) :
        Safety.traceSafeList M bounds [stmt] state ↔ Safety.stepSafe M bounds stmt state := by
      rw [Safety.traceSafeList_cons]
      simp
    cases phase <;>
      simp [chunk,ComputeKernel.surfaceBody,List.take_succ_cons,last,
        Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
        Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
        Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
        Safety.maskActive_unfold,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
        ComputeDType.eraseDType,numeric,bop,natLt,numericLt,Region.cast,ofFloat,toFloat,
        h.row]
    all_goals exact fun j => hx (k*B+j.val)
  · intro state _
    apply list_safe
    intro stmt member before
    cases phase
    all_goals
      simp only [chunk,ComputeKernel.surfaceBody,Bool.false_eq_true,↓reduceIte,
        List.drop_succ_cons,List.drop_zero,List.mem_cons,List.not_mem_nil,or_false] at member
    · rcases member with rfl | rfl | rfl | rfl
      all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]
    · rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl
      all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]

theorem chunks_safe (M : Algebra α) (phase : Bool) (s t : State α)
    (X : RegionName) (D B start count : Nat) (values : Fin B → α) (mean : α)
    (h : Ready s t values mean) (bounds : RegionName → Nat)
    (hx : ∀ j, j < D → j+D*s.pids 0 < bounds X) :
    Safety.traceSafeList M bounds (chunks phase X D B start count) t := by
  induction count generalizing start t values with
  | zero => simp [chunks]
  | succ count ih =>
    obtain ⟨u,hu,hr⟩ := chunk_run M phase s t X D B start values mean h
    exact append_safe _ _ _ _ _ _ (chunk_safe M phase s t X D B start values mean h bounds hx)
      hu (ih u (start+1) _ hr)

theorem tree_safe (M : Algebra α) (s : State α) (bounds : RegionName → Nat)
    {B : Nat} (tree : ReductionTree B) :
    Safety.safeOp M (some .fp32) bounds s (sumExpr tree) := by
  induction tree with
  | leaf i =>
    simp only [sumExpr]
    rw [Safety.safeOp_unfold]
    dsimp only
    rw [Safety.safeOp_unfold]
    trivial
  | node l r hl hr =>
    simp only [sumExpr]
    rw [Safety.safeOp_unfold]
    dsimp only
    exact ⟨hl,hr⟩

theorem sum_safe (M : Algebra α) (s : State α) (bounds : RegionName → Nat)
    {B : Nat} (tree : ReductionTree B) :
    Safety.traceSafeList M bounds (sumCode tree) s := by
  apply Safety.traceSafeList_cons_intro
  · rw [Safety.stepSafe_unfold]
    dsimp only
    rw [Safety.safeExpr_unfold]
    dsimp only
    rw [Safety.safeCompute_unfold]
    dsimp only
    rw [Safety.safeOp_unfold]
    dsimp only
    exact tree_safe M s bounds tree
  · intro t _
    exact Safety.traceSafeList_nil M bounds t

theorem mean_safe (M : Algebra α) (s t : State α) (Mean : RegionName) (D : Nat)
    (mean total : α) (h : Summed s t mean total) (bounds : RegionName → Nat)
    (hm : s.pids 0 < bounds Mean) : Safety.traceSafeList M bounds (meanCode Mean D) t := by
  simp [meanCode,ComputeKernel.surfaceBody,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
    Safety.maskActive_unfold,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,numeric,bop,Region.cast,ofFloat,toFloat,h.total,h.row,
    store,TileShape.allIndices,List.finRange_succ,Fin.eq_zero,hm]

theorem inverse_safe (M : Algebra α) (s t : State α) (Rstd : RegionName) (D : Nat)
    (mean total : α) (h : Summed s t mean total) (bounds : RegionName → Nat)
    (hr : s.pids 0 < bounds Rstd) : Safety.traceSafeList M bounds (inverseCode Rstd D) t := by
  simp [inverseCode,ComputeKernel.surfaceBody,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
    Safety.maskActive_unfold,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,numeric,bop,Region.cast,ofFloat,toFloat,h.total,h.row,
    store,TileShape.allIndices,List.finRange_succ,Fin.eq_zero,hr]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormWelfold.SourceSafety
