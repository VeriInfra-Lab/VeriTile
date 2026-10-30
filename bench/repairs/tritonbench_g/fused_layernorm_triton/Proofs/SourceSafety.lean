import bench.repairs.tritonbench_g.fused_layernorm_triton.Proofs.Kernel
import VeriTile.Triton.Float.StructuralSafety
import bench.repairs.tritonbench_g.fused_layernorm_triton.Proofs.ReadableSafety

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

theorem prefix_safe (M : Algebra α) (eq32 : α → α → Bool)
    (s : State α) (X : RegionName) (S D B : Nat) (bounds : RegionName → Nat)
    (hx : ∀ j,j<D → j+D*s.pids 0<bounds X) :
    Safety.traceSafeList (withEquality M eq32) bounds (welfordPrefix X S D B) s := by
  simp [welfordPrefix,ComputeKernel.surfaceBody,
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,
    Safety.addressSafe,Safety.maskActive,
    FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,
    withEquality,Region.cast,ofFloat,toFloat]
  exact fun j => hx j.val


theorem chunk_safe (M : Algebra α) (eq32 : α → α → Bool)
    (s t : State α) (X : RegionName) (D B k : Nat)
    (previous : Fin B → WelfordTriple α) (h : WelfordLanesReady s t previous)
    (bounds : RegionName → Nat) (hx : ∀ j,j<D → j+D*s.pids 0<bounds X) :
    Safety.traceSafeList (withEquality M eq32) bounds (Whole.nextChunk X D B k) t := by
  simp [Whole.nextChunk,ComputeKernel.surfaceBody,
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,
    Safety.addressSafe,Safety.maskActive,
    FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,
    withEquality,Region.cast,ofFloat,toFloat,h.mean,h.m2,h.weight,h.row,h.mem]
  exact fun j => hx (k*B+j.val)


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

theorem chunks_safe (M : Algebra α) (eq32 : α → α → Bool)
    (s t : State α) (X : RegionName) (D B start count : Nat)
    (previous : Fin B → WelfordTriple α) (h : WelfordLanesReady s t previous)
    (bounds : RegionName → Nat) (hx : ∀ j,j<D → j+D*s.pids 0<bounds X) :
    Safety.traceSafeList (withEquality M eq32) bounds (Whole.chunkCode X D B start count) t := by
  induction count generalizing start t previous with
  | zero => simp [Whole.chunkCode]
  | succ count ih =>
    obtain ⟨u,hu,hr⟩ := next_chunk_run M eq32 s t X D B start previous h
    exact append_safe _ _ _ _ _ _ (chunk_safe M eq32 s t X D B start previous h bounds hx)
      hu (ih u (start+1) _ hr)

theorem statistics_safe (M : Algebra α) (eq32 : α → α → Bool)
    (s : State α) (X Mean Rstd : RegionName) (S D B : Nat) (tree : ReductionTree B)
    (bounds : RegionName → Nat) (hx : ∀ j,j<D → j+D*s.pids 0<bounds X)
    (bm : s.pids 0<bounds Mean) (br : s.pids 0<bounds Rstd) :
    TraceSafe (withEquality M eq32) bounds (statisticsKernel X Mean Rstd S D B tree) s := by
  obtain ⟨p,hp,hpr⟩ := welford_prefix_run M eq32 s X S D B
  obtain ⟨q,hq,hqr⟩ := chunks_run M eq32 s p X D B 1 (blockCount D B-1) _ hpr
  obtain ⟨u,hu,hur⟩ := welford_tuple_run M eq32 s q _ hqr tree
  unfold TraceSafe statisticsKernel
  simp only [ComputeKernel.surfaceBody,List.append_assoc]
  exact append_safe _ _ _ _ _ _ (prefix_safe M eq32 s X S D B bounds hx) hp
    (append_safe _ _ _ _ _ _ (chunks_safe M eq32 s p X D B 1 (blockCount D B-1) _ hpr bounds hx) hq
      (append_safe _ _ _ _ _ _ (tuple_safe (withEquality M eq32) bounds q tree) hu
        (suffix_safe M eq32 s u Mean Rstd D _ hur bounds bm br)))

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
