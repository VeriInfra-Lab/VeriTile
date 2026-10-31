import bench.tritonbench_g.fused_layernorm_triton.Proofs.Kernel
import VeriTile.Triton.Float.StructuralSafety
import bench.tritonbench_g.fused_layernorm_triton.Proofs.ReadableSafety

/-! The actual source access walk reuses the established Welford and
normalization execution contexts. Only active source addresses need backing. -/
namespace VeriTile.Bench.Repairs.FusedLayernormTriton.SourceSafety
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.FusedLayernormTriton FPStages Whole
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem normalize_safe (M : Algebra α) (eq32 : α → α → Bool)
    (s : State α) (X W Bias Y : RegionName) (D B k row : Nat) (mean inverse : α)
    (hm : s.regs .real [1,1] "tmp3"=some (fun _ => mean))
    (hr : s.regs .real [1,1] "tmp10"=some (fun _ => inverse))
    (hx : s.regs .nat [1,1] "x0"=some (fun _ => row))
    (bounds : RegionName → Nat)
    (bx : ∀ j,j<D → j+D*row<bounds X)
    (bw : ∀ j,j<D → j<bounds W)
    (bb : ∀ j,j<D → j<bounds Bias)
    (byOut : ∀ j,j<D → j+D*row<bounds Y) :
    Safety.traceSafeList (withEquality M eq32) bounds (Whole.normalizeChunk X W Bias Y D B k) s := by
  simp [Whole.normalizeChunk,ComputeKernel.surfaceBody,
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,
    Safety.addressSafe,Safety.maskActive,
    FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,
    withEquality,Region.cast,ofFloat,toFloat,hm,hr,hx,store]

  exact ⟨fun j => bx (k*B+j.val),fun j => bw (k*B+j.val),
    fun j => bb (k*B+j.val),fun j => byOut (k*B+j.val)⟩

private theorem list_safe (M : Algebra α) (bounds : RegionName → Nat)
    (code : List ComputeStmt)
    (each : ∀ st∈code,∀ s : State α,Safety.stepSafe M bounds st s) (s : State α) :
    Safety.traceSafeList M bounds code s := by
  induction code generalizing s with
  | nil => simp [Safety.traceSafeList]
  | cons head tail ih =>
    apply Safety.traceSafeList_cons_intro (each head (by simp) s)
    intro t _
    exact ih (fun st hst => each st (by simp [hst])) t

theorem tree_safe (M : Algebra α) (bounds : RegionName → Nat)
    (s : State α) {B : Nat} (tree : ReductionTree B) :
    Safety.safeOp M (some .fp32) bounds s (FPStages.treeExpressions tree).mean ∧
    Safety.safeOp M (some .fp32) bounds s (FPStages.treeExpressions tree).m2 ∧
    Safety.safeOp M (some .fp32) bounds s (FPStages.treeExpressions tree).weight := by
  induction tree with
  | leaf i => simp [FPStages.treeExpressions,Safety.safeOp]
  | node l r hl hr =>
    simp [FPStages.treeExpressions,FPStages.combineExpressions,Safety.safeOp,
      hl.1,hl.2.1,hl.2.2,hr.1,hr.2.1,hr.2.2]

theorem tuple_safe (M : Algebra α) (bounds : RegionName → Nat)
    (s : State α) {B : Nat} (tree : ReductionTree B) :
    Safety.traceSafeList M bounds (welfordTupleCode tree) s := by
  apply list_safe
  intro st member t
  simp only [welfordTupleCode,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl | rfl | rfl
  · simpa [Safety.stepSafe_unfold,Safety.safeExpr,Safety.safeCompute,Safety.safeOp]
      using (tree_safe M bounds t tree).1
  · simpa [Safety.stepSafe_unfold,Safety.safeExpr,Safety.safeCompute,Safety.safeOp]
      using (tree_safe M bounds t tree).2.1
  · simpa [Safety.stepSafe_unfold,Safety.safeExpr,Safety.safeCompute,Safety.safeOp]
      using (tree_safe M bounds t tree).2.2

theorem suffix_safe (M : Algebra α) (eq32 : α → α → Bool)
    (s t : State α) (Mean Rstd : RegionName) (D : Nat)
    (v : WelfordTriple α) (h : WelfordTupleReady s t v)
    (bounds : RegionName → Nat) (bm : s.pids 0<bounds Mean) (br : s.pids 0<bounds Rstd) :
    Safety.traceSafeList (withEquality M eq32) bounds (welfordSuffix Rstd Mean D) t := by
  simp [welfordSuffix,ComputeKernel.surfaceBody,
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,
    Safety.addressSafe,Safety.maskActive,
    FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,bop,store,
    withEquality,Region.cast,ofFloat,toFloat,h.mean,h.m2,h.weight,h.row,
    TileShape.allIndices,List.finRange_succ,Fin.eq_zero,bm,br]

private theorem append_safe (M : Algebra α) (bounds : RegionName → Nat)
    (a b : List ComputeStmt) (s t : State α)
    (ha : Safety.traceSafeList M bounds a s) (ran : run M a s=some t)
    (hb : Safety.traceSafeList M bounds b t) :
    Safety.traceSafeList M bounds (a++b) s := by
  apply (Safety.traceSafeList_append M bounds a b s).mpr
  refine ⟨ha,?_⟩
  intro u hu
  have eq : t=u := Option.some.inj (ran.symm.trans hu)
  simpa only [←eq] using hb

theorem normalize_loop_safe (M : Algebra α) (eq32 : α → α → Bool)
    (s : State α) (X W Bias Y : RegionName) (D B start count row : Nat) (mean inverse : α)
    (hm : s.regs .real [1,1] "tmp3"=some (fun _ => mean))
    (hr : s.regs .real [1,1] "tmp10"=some (fun _ => inverse))
    (hx : s.regs .nat [1,1] "x0"=some (fun _ => row))
    (bounds : RegionName → Nat)
    (bx : ∀ j,j<D → j+D*row<bounds X)
    (bw : ∀ j,j<D → j<bounds W)
    (bb : ∀ j,j<D → j<bounds Bias)
    (byOut : ∀ j,j<D → j+D*row<bounds Y) :
    Safety.traceSafeList (withEquality M eq32) bounds
      (Whole.normalizeCode X W Bias Y D B start count) s := by
  induction count generalizing s start with
  | zero => simp [Whole.normalizeCode]
  | succ count ih =>
    obtain ⟨u,hu,_,hreg⟩ := normalize_chunk_run M eq32 s X W Bias Y D B start row mean inverse hm hr hx
    exact append_safe _ _ _ _ _ _
      (normalize_safe M eq32 s X W Bias Y D B start row mean inverse hm hr hx bounds bx bw bb byOut)
      hu (ih u (start+1) hreg.mean hreg.rstd hreg.row)

theorem source_safe (M : Algebra α) (eq32 : α → α → Bool)
    (s : State α) (X W Bias Y Mean Rstd : RegionName) (S D : Nat) (c : Contract.Configuration)
    (bounds : RegionName → Nat)
    (bx : ∀ j,j<D → j+D*s.pids 0<bounds X)
    (bw : ∀ j,j<D → j<bounds W)
    (bb : ∀ j,j<D → j<bounds Bias)
    (byOut : ∀ j,j<D → j+D*s.pids 0<bounds Y)
    (bm : s.pids 0<bounds Mean) (br : s.pids 0<bounds Rstd) :
    TraceSafe (withEquality M eq32) bounds (Contract.source X W Bias Y Mean Rstd S D c) s := by
  exact ReadableSafety.source_safe_for Bool.false M eq32 s X W Bias Y Mean Rstd S D c.block c.tree
    c.positive bounds bx bw bb byOut bm br

#axiomsClean source_safe

end
end VeriTile.Bench.Repairs.FusedLayernormTriton.SourceSafety
