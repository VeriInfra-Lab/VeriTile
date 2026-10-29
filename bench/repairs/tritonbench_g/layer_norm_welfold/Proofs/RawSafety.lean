import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.RawGrid
namespace VeriTile.Bench.Repairs.LayerNormWelfold.Raw
open VeriTile.Triton FP.Structural Reuse
noncomputable section
set_option maxHeartbeats 3000000
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

theorem tree_positive {B : Nat} (tree : ReductionTree B) : 0<B := by
  induction tree with
  | leaf i => exact Nat.lt_of_le_of_lt (Nat.zero_le i.val) i.isLt
  | node l r hl hr => exact hl

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

theorem init_safe (M : Algebra α) (s : State α) (S D B : Nat)
    (tree : DSL.WelfordSource.Tree B) (bounds : RegionName → Nat) :
    Safety.traceSafeList M bounds (initBody S D B tree) s := by
  apply list_safe
  intro st member t
  simp only [initBody,code,SourceCode.rowKernel,ComputeKernel.surfaceBody,List.take,
    List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl
  all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]

theorem reset_safe (M : Algebra α) (s : State α) (S D B : Nat)
    (tree : DSL.WelfordSource.Tree B) (bounds : RegionName → Nat) :
    Safety.traceSafeList M bounds (reset S D B tree) s := by
  apply list_safe
  intro st member t
  simp only [reset,code,SourceCode.rowKernel,ComputeKernel.surfaceBody,List.drop,List.take,
    List.mem_singleton] at member
  subst st
  simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]

theorem tree_safe {B : Nat} (M : Algebra α) (s : State α) (name : RegName)
    (tree : ReductionTree B) (bounds : RegionName → Nat) :
    Safety.safeOp M (some .fp32) bounds s
      (DSL.WelfordSource.sumExpression (.ref .real [1,B] name) (Readable.tree tree)) := by
  induction tree with
  | leaf i =>
    simp only [Readable.tree,DSL.WelfordSource.sumExpression]
    rw [Safety.safeOp_unfold]
    dsimp only
    rw [Safety.safeOp_unfold]
    trivial
  | node l r hl hr =>
    simp only [Readable.tree,DSL.WelfordSource.sumExpression]
    rw [Safety.safeOp_unfold]
    exact ⟨hl,hr⟩


theorem sum_safe (M : Algebra α) (phase : Bool) (s : State α) (S D B : Nat)
    (bounds : RegionName → Nat) (tree : ReductionTree B) :
    Safety.traceSafeList M bounds (sumCode phase S D B (Readable.tree tree)) s := by
  apply list_safe
  intro st member before
  cases phase
  all_goals
    simp only [sumCode,code,SourceCode.rowKernel,ComputeKernel.surfaceBody,
      Bool.false_eq_true,↓reduceIte,List.drop,List.take,List.mem_singleton] at member
    subst st
    simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold]
    rw [Safety.safeOp_unfold]
    dsimp only
    rw [Safety.safeOp_unfold]
    dsimp only
    rw [Safety.safeOp_unfold]
    dsimp only
    exact tree_safe M before _ tree bounds


theorem phase_safe (M : Algebra α) (phase : Bool) (s t : State α)
    (X : RegionName) (S D B k : Nat) (rt : ReductionTree B)
    (values : Fin B → α) (mean : α) (h : Ready phase s t values mean)
    (bounds : RegionName → Nat) (hx : ∀ j, j<D → j+D*s.pids 0<bounds X) :
    Safety.traceSafeList M bounds (phaseBody phase X S D B (Readable.tree rt))
      (t.setReg "roffset" .nat [] (fun _ => k*B))  := by
  have hv := h.values
  cases phase <;> dsimp at hv
  all_goals
    simp [phaseBody,code,SourceCode.rowKernel,ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
      FP.Structural.numeric,FP.Structural.bop,FP.Structural.natLt,FP.Structural.numericLt,
      Region.cast,ofFloat,toFloat,h.row,h.mean,hv,h.base]
  all_goals exact fun (j : Fin B) => hx (k*B+j.val)

theorem phase_loop_safe (M : Algebra α) (phase : Bool) (s t : State α)
    (X : RegionName) (S D B start count : Nat) (hB : 0<B)
    (count_eq : start+count=blockCount D B) (rt : ReductionTree B)
    (values : Fin B → α) (mean : α) (h : Ready phase s t values mean)
    (bounds : RegionName → Nat) (hx : ∀ j, j<D → j+D*s.pids 0<bounds X) :
    Safety.rangeSafe M bounds "roffset" (start*B) D B (phaseBody phase X S D B (Readable.tree rt)) t := by
  induction count generalizing start t values with
  | zero =>
    have stopped : ¬start*B<D := by
      intro active
      have := (range_offset_iff D B start hB).mpr active
      omega
    simp [Safety.rangeSafe_unfold,stopped]
  | succ count ih =>
    obtain ⟨u,hu,hp⟩ := phase_run M phase s t X S D B start (Readable.tree rt) values mean h
    have active : start*B<D := (range_offset_iff D B start hB).mp (by omega)
    rw [Safety.rangeSafe_unfold,if_neg (Nat.ne_of_gt hB),if_pos active]
    refine ⟨phase_safe M phase s t X S D B start rt values mean h bounds hx,?_⟩
    rw [hu]
    simpa only [show start*B+B=(start+1)*B by simp [Nat.add_mul]] using ih u (start+1) (by omega) _ hp

theorem phase_for_safe (M : Algebra α) (phase : Bool) (s t : State α)
    (X : RegionName) (S D B : Nat) (rt : ReductionTree B)
    (values : Fin B → α) (mean : α) (h : Ready phase s t values mean)
    (bounds : RegionName → Nat) (hx : ∀ j, j<D → j+D*s.pids 0<bounds X) :
    Safety.traceSafeList M bounds [.forRange "roffset" 0 D B (phaseBody phase X S D B (Readable.tree rt))] t := by
  apply Safety.traceSafeList_cons_intro
  · rw [Safety.stepSafe_unfold]
    simpa only [Nat.zero_mul] using phase_loop_safe M phase s t X S D B 0 (blockCount D B) (tree_positive rt) (by simp) rt values mean h bounds hx
  · intro u _; exact Safety.traceSafeList_nil _ _ _
theorem mean_safe (M : Algebra α) (s t : State α) (Mean : RegionName) (S D B : Nat) (rt : ReductionTree B)
    (mean total : α) (h : Summed Bool.false s t B mean total) (bounds : RegionName → Nat)
    (hm : s.pids 0 < bounds Mean) : Safety.traceSafeList M bounds (meanTail Mean S D B (Readable.tree rt)) t := by
  have htotal := h.total
  dsimp at htotal
  simp [meanTail,code,SourceCode.rowKernel,ComputeKernel.surfaceBody,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
    Safety.maskActive_unfold,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,FP.Structural.numeric,FP.Structural.bop,Region.cast,ofFloat,toFloat,htotal,h.row,
    store,TileShape.allIndices,List.finRange_succ,Fin.eq_zero,hm]


theorem inverse_safe (M : Algebra α) (s t : State α) (Rstd : RegionName) (S D B : Nat) (rt : ReductionTree B)
    (mean total : α) (h : Summed Bool.true s t B mean total) (bounds : RegionName → Nat)
    (hr : s.pids 0 < bounds Rstd) : Safety.traceSafeList M bounds (inverseTail Rstd S D B (Readable.tree rt)) t := by
  have htotal := h.total
  dsimp at htotal
  simp [inverseTail,code,SourceCode.rowKernel,ComputeKernel.surfaceBody,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
    Safety.maskActive_unfold,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,FP.Structural.numeric,FP.Structural.bop,Region.cast,ofFloat,toFloat,htotal,h.row,
    store,TileShape.allIndices,List.finRange_succ,Fin.eq_zero,hr]


theorem stats_safe (M : Algebra α) (s : State α) (X Mean Rstd : RegionName)
    (S D B : Nat) (rt : ReductionTree B) (bounds : RegionName → Nat)
    (hx : ∀ j, j<D → j+D*s.pids 0<bounds X)
    (hm : s.pids 0<bounds Mean) (hr : s.pids 0<bounds Rstd) :
    Safety.traceSafeList M bounds (stats X Mean Rstd S D B (Readable.tree rt)) s := by
  obtain ⟨a,ha,hia⟩ := prefix_run M s S D B (Readable.tree rt)
  obtain ⟨b,hb,hib⟩ := phase_loop M Bool.false s a X S D B 0 (blockCount D B) (Readable.tree rt) (tree_positive rt) (by simp) _ _ hia
  obtain ⟨c,hc,hic⟩ := sum_run M Bool.false s b S D B rt _ _ hib
  obtain ⟨d,hd,hid⟩ := mean_run M s c Mean S D B (Readable.tree rt) _ _ hic
  obtain ⟨e,he,hie⟩ := reset_mean_run M s d Mean S D B (Readable.tree rt) _ hid
  obtain ⟨f,hf,hif⟩ := phase_loop M Bool.true _ e X S D B 0 (blockCount D B) (Readable.tree rt) (tree_positive rt) (by simp) _ _ hie
  obtain ⟨g,hg,hig⟩ := sum_run M Bool.true _ f S D B rt _ _ hif
  simp only [Nat.zero_mul] at hb hf
  have hbr : run M [.forRange "roffset" 0 D B (phaseBody Bool.false X S D B (Readable.tree rt))] a=some b := by
    simp [run,step,hb]
  have hfr : run M [.forRange "roffset" 0 D B (phaseBody Bool.true X S D B (Readable.tree rt))] e=some f := by
    simp [run,step,hf]
  rw [stats_split]
  simp only [List.append_assoc]
  exact append_safe _ _ _ _ _ _ (init_safe M s S D B (Readable.tree rt) bounds) ha
    (append_safe _ _ _ _ _ _ (phase_for_safe M Bool.false s a X S D B rt _ _ hia bounds hx) hbr
      (append_safe _ _ _ _ _ _ (sum_safe M Bool.false b S D B bounds rt) hc
        (append_safe _ _ _ _ _ _ (mean_safe M s c Mean S D B rt _ _ hic bounds hm) hd
          (append_safe _ _ _ _ _ _ (reset_safe M d S D B (Readable.tree rt) bounds) he
            (append_safe _ _ _ _ _ _ (phase_for_safe M Bool.true _ e X S D B rt _ _ hie bounds hx) hfr
              (append_safe _ _ _ _ _ _ (sum_safe M Bool.true f S D B bounds rt) hg
                (inverse_safe M _ g Rstd S D B rt _ _ hig bounds hr)))))))

theorem normalize_safe (M : Algebra α) (eq32 : α → α → Bool)
    (s : State α) (X W Bias Y : RegionName) (S D B k row : Nat)
    (rt : ReductionTree B) (reuse : Bool) (mean inverse : α)
    (hm : s.regs .real [1,1] "tmp6"=some (fun _ => mean))
    (hr : s.regs .real [1,1] "tmp18"=some (fun _ => inverse))
    (hx : s.regs .nat [1,1] "x0"=some (fun _ => row))
    (hb : s.regs .nat [1,B] "rbase"=some (fun i => i.2.1.val))
    (bounds : RegionName → Nat)
    (bx : ∀ j,j<D → j+D*row<bounds X)
    (bw : ∀ j,j<D → j<bounds W) (bb : ∀ j,j<D → j<bounds Bias)
    (byOut : ∀ j,j<D → j+D*row<bounds Y) :
    Safety.traceSafeList (withEquality M eq32) bounds (normalizeBody reuse X W Bias Y S D B (Readable.tree rt))
      (s.setReg "roffset" .nat [] (fun _ => k*B)) := by
  cases reuse <;>
    simp [normalizeBody,SourceCode.selected,SourceCode.rowKernel,SourceCode.rowOffsetReusedKernel,
      ComputeKernel.surfaceBody,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
      Safety.maskActive_unfold,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      ComputeDType.eraseDType,FP.Structural.numeric,FP.Structural.bop,FP.Structural.natLt,
      FP.Structural.numericLt,withEquality,Region.cast,ofFloat,toFloat,hm,hr,hx,hb,store]
  all_goals exact ⟨fun j => bx (k*B+j.val), fun j => bw (k*B+j.val),
    fun j => bb (k*B+j.val), fun j => byOut (k*B+j.val)⟩

theorem normalize_loop_safe (M : Algebra α) (eq32 : α → α → Bool)
    (s : State α) (X W Bias Y : RegionName) (S D B start count row : Nat) (hB : 0<B)
    (count_eq : start+count=blockCount D B) (rt : ReductionTree B) (reuse : Bool) (mean inverse : α)
    (hm : s.regs .real [1,1] "tmp6"=some (fun _ => mean))
    (hr : s.regs .real [1,1] "tmp18"=some (fun _ => inverse))
    (hx : s.regs .nat [1,1] "x0"=some (fun _ => row))
    (hb : s.regs .nat [1,B] "rbase"=some (fun i => i.2.1.val))
    (bounds : RegionName → Nat)
    (bx : ∀ j,j<D → j+D*row<bounds X)
    (bw : ∀ j,j<D → j<bounds W) (bb : ∀ j,j<D → j<bounds Bias)
    (byOut : ∀ j,j<D → j+D*row<bounds Y) :
    Safety.rangeSafe (withEquality M eq32) bounds "roffset" (start*B) D B
      (normalizeBody reuse X W Bias Y S D B (Readable.tree rt)) s := by
  induction count generalizing s start with
  | zero =>
    have stopped : ¬start*B<D := by
      intro active
      have := (range_offset_iff D B start hB).mpr active
      omega
    simp [Safety.rangeSafe_unfold,stopped]
  | succ count ih =>
    obtain ⟨u,hu,_,hreg⟩ := normalize_chunk_run M eq32 s X W Bias Y S D B start row rt reuse mean inverse hm hr hx hb
    have active : start*B<D := (range_offset_iff D B start hB).mp (by omega)
    rw [Safety.rangeSafe_unfold,if_neg (Nat.ne_of_gt hB),if_pos active]
    refine ⟨normalize_safe M eq32 s X W Bias Y S D B start row rt reuse mean inverse hm hr hx hb bounds bx bw bb byOut,?_⟩
    rw [hu]
    simpa only [show start*B+B=(start+1)*B by simp [Nat.add_mul]] using ih u (start+1) (by omega) hreg.mean hreg.rstd hreg.row hreg.base

theorem source_safe (M : Algebra α) (eq32 : α → α → Bool) (reuse : Bool)
    (s : State α) (X W Bias Y Mean Rstd : RegionName) (S D B : Nat) (rt : ReductionTree B)
    (sep : X ≠ Mean) (bounds : RegionName → Nat)
    (bx : ∀ j,j<D → j+D*s.pids 0<bounds X)
    (bw : ∀ j,j<D → j<bounds W) (bb : ∀ j,j<D → j<bounds Bias)
    (byOut : ∀ j,j<D → j+D*s.pids 0<bounds Y)
    (bm : s.pids 0<bounds Mean) (br : s.pids 0<bounds Rstd) :
    TraceSafe (withEquality M eq32) bounds
      (SourceCode.selected reuse X W Bias Y Mean Rstd S D B (Readable.tree rt)) s := by
  obtain ⟨t,ran,ready⟩ := stats_run (withEquality M eq32) s X Mean Rstd S D B rt (tree_positive rt) sep
  unfold TraceSafe
  rw [source_split]
  apply append_safe _ _ _ _ _ _ (stats_safe (withEquality M eq32) s X Mean Rstd S D B rt bounds bx bm br) ran
  apply Safety.traceSafeList_cons_intro
  · rw [Safety.stepSafe_unfold]
    simpa only [Nat.zero_mul] using normalize_loop_safe M eq32 t X W Bias Y S D B 0 (blockCount D B) (s.pids 0) (tree_positive rt) (by simp) rt reuse _ _
      ready.mean ready.inverse ready.row ready.base bounds bx bw bb byOut
  · intro u _; exact Safety.traceSafeList_nil _ _ _

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.LayerNormWelfold.Raw
