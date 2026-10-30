import bench.tritonbench_g.layer_norm_welfold.Proofs.ReadableGrid
namespace VeriTile.Bench.Repairs.LayerNormWelfold.Readable
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

theorem phase_safe (M : Algebra α) (phase : Bool) (s t : State α)
    (X : RegionName) (S D B k : Nat) (rt : ReductionTree B)
    (values : Fin B → α) (mean : α) (h : Ready s t values mean)
    (bounds : RegionName → Nat) (hx : ∀ j, j<D → j+D*s.pids 0<bounds X) :
    Safety.traceSafeList M bounds (phaseBody phase X S D B rt)
      (t.setReg "k" .nat [] (fun _ => k)) := by
  cases phase <;>
    simp [phaseBody,code,SourceCode.rowKernel,ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
      FP.Structural.numeric,FP.Structural.bop,FP.Structural.natLt,FP.Structural.numericLt,
      Region.cast,ofFloat,toFloat,h.row,h.mean,h.values]
  all_goals exact fun (j : Fin B) => hx (k*B+j.val)

theorem phase_loop_safe (M : Algebra α) (phase : Bool) (s t : State α)
    (X : RegionName) (S D B start count : Nat) (rt : ReductionTree B)
    (values : Fin B → α) (mean : α) (h : Ready s t values mean)
    (bounds : RegionName → Nat) (hx : ∀ j, j<D → j+D*s.pids 0<bounds X) :
    Safety.loopSafe M bounds "k" start (start+count) (phaseBody phase X S D B rt) t := by
  induction count generalizing start t values with
  | zero => simp [Safety.loopSafe_unfold]
  | succ count ih =>
    obtain ⟨u,hu,hp⟩ := phase_run M phase s t X S D B start rt values mean h
    rw [Safety.loopSafe_unfold,if_pos (by omega)]
    refine ⟨phase_safe M phase s t X S D B start rt values mean h bounds hx,?_⟩
    rw [hu]
    simpa only [show start+Nat.succ count=(start+1)+count by omega] using ih u (start+1) _ hp

theorem phase_for_safe (M : Algebra α) (phase : Bool) (s t : State α)
    (X : RegionName) (S D B : Nat) (rt : ReductionTree B)
    (values : Fin B → α) (mean : α) (h : Ready s t values mean)
    (bounds : RegionName → Nat) (hx : ∀ j, j<D → j+D*s.pids 0<bounds X) :
    Safety.traceSafeList M bounds [.forLoop "k" (blockCount D B) (phaseBody phase X S D B rt)] t := by
  apply Safety.traceSafeList_cons_intro
  · rw [Safety.stepSafe_unfold]
    simpa only [Nat.zero_add] using phase_loop_safe M phase s t X S D B 0 (blockCount D B) rt values mean h bounds hx
  · intro u _; exact Safety.traceSafeList_nil _ _ _
theorem mean_safe (M : Algebra α) (s t : State α) (Mean : RegionName) (S D B : Nat) (rt : ReductionTree B)
    (mean total : α) (h : Summed s t mean total) (bounds : RegionName → Nat)
    (hm : s.pids 0 < bounds Mean) : Safety.traceSafeList M bounds (meanTail Mean S D B rt) t := by
  simp [meanTail,code,SourceCode.rowKernel,ComputeKernel.surfaceBody,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
    Safety.maskActive_unfold,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,FP.Structural.numeric,FP.Structural.bop,Region.cast,ofFloat,toFloat,h.total,h.row,
    store,TileShape.allIndices,List.finRange_succ,Fin.eq_zero,hm]


theorem inverse_safe (M : Algebra α) (s t : State α) (Rstd : RegionName) (S D B : Nat) (rt : ReductionTree B)
    (mean total : α) (h : Summed s t mean total) (bounds : RegionName → Nat)
    (hr : s.pids 0 < bounds Rstd) : Safety.traceSafeList M bounds (inverseTail Rstd S D B rt) t := by
  simp [inverseTail,code,SourceCode.rowKernel,ComputeKernel.surfaceBody,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
    Safety.maskActive_unfold,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,FP.Structural.numeric,FP.Structural.bop,Region.cast,ofFloat,toFloat,h.total,h.row,
    store,TileShape.allIndices,List.finRange_succ,Fin.eq_zero,hr]


theorem stats_safe (M : Algebra α) (s : State α) (X Mean Rstd : RegionName)
    (S D B : Nat) (rt : ReductionTree B) (bounds : RegionName → Nat)
    (hx : ∀ j, j<D → j+D*s.pids 0<bounds X)
    (hm : s.pids 0<bounds Mean) (hr : s.pids 0<bounds Rstd) :
    Safety.traceSafeList M bounds (stats X Mean Rstd S D B rt) s := by
  obtain ⟨a,ha,hia⟩ := init_run M s S B
  obtain ⟨b,hb,hib⟩ := phase_loop M Bool.false s a X S D B 0 (blockCount D B) rt _ _ hia
  obtain ⟨c,hc,hic⟩ := sum_run M s b _ _ hib rt
  obtain ⟨d,hd,hid⟩ := mean_run M s c Mean S D B rt _ _ hic
  obtain ⟨e,he,hie⟩ := reset_mean_run M s d Mean B _ hid
  obtain ⟨f,hf,hif⟩ := phase_loop M Bool.true _ e X S D B 0 (blockCount D B) rt _ _ hie
  obtain ⟨g,hg,hig⟩ := sum_run M _ f _ _ hif rt
  simp only [Nat.zero_add] at hb hf
  have hbr : run M [.forLoop "k" (blockCount D B) (phaseBody Bool.false X S D B rt)] a=some b := by
    simp [run,step,hb]
  have hfr : run M [.forLoop "k" (blockCount D B) (phaseBody Bool.true X S D B rt)] e=some f := by
    simp [run,step,hf]
  rw [stats_split]
  simp only [List.append_assoc]
  exact append_safe _ _ _ _ _ _ (SourceSafety.init_safe M s S B bounds) ha
    (append_safe _ _ _ _ _ _ (phase_for_safe M Bool.false s a X S D B rt _ _ hia bounds hx) hbr
      (append_safe _ _ _ _ _ _ (SourceSafety.sum_safe M b bounds rt) hc
        (append_safe _ _ _ _ _ _ (mean_safe M s c Mean S D B rt _ _ hic bounds hm) hd
          (append_safe _ _ _ _ _ _ (SourceSafety.reset_safe M d B bounds) he
            (append_safe _ _ _ _ _ _ (phase_for_safe M Bool.true _ e X S D B rt _ _ hie bounds hx) hfr
              (append_safe _ _ _ _ _ _ (SourceSafety.sum_safe M f bounds rt) hg
                (inverse_safe M _ g Rstd S D B rt _ _ hig bounds hr)))))))

theorem normalize_safe (M : Algebra α) (eq32 : α → α → Bool)
    (s : State α) (X W Bias Y : RegionName) (S D B k row : Nat)
    (rt : ReductionTree B) (reuse : Bool) (mean inverse : α)
    (hm : s.regs .real [1,1] "tmp3"=some (fun _ => mean))
    (hr : s.regs .real [1,1] "tmp10"=some (fun _ => inverse))
    (hx : s.regs .nat [1,1] "x0"=some (fun _ => row))
    (bounds : RegionName → Nat)
    (bx : ∀ j,j<D → j+D*row<bounds X)
    (bw : ∀ j,j<D → j<bounds W) (bb : ∀ j,j<D → j<bounds Bias)
    (byOut : ∀ j,j<D → j+D*row<bounds Y) :
    Safety.traceSafeList (withEquality M eq32) bounds (normalizeBody reuse X W Bias Y S D B rt)
      (s.setReg "k" .nat [] (fun _ => k)) := by
  cases reuse <;>
    simp [normalizeBody,SourceCode.selected,SourceCode.rowKernel,SourceCode.rowOffsetReusedKernel,
      ComputeKernel.surfaceBody,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
      Safety.maskActive_unfold,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      ComputeDType.eraseDType,FP.Structural.numeric,FP.Structural.bop,FP.Structural.natLt,
      FP.Structural.numericLt,withEquality,Region.cast,ofFloat,toFloat,hm,hr,hx,store]
  all_goals exact ⟨fun j => bx (k*B+j.val), fun j => bw (k*B+j.val),
    fun j => bb (k*B+j.val), fun j => byOut (k*B+j.val)⟩

theorem normalize_loop_safe (M : Algebra α) (eq32 : α → α → Bool)
    (s : State α) (X W Bias Y : RegionName) (S D B start count row : Nat)
    (rt : ReductionTree B) (reuse : Bool) (mean inverse : α)
    (hm : s.regs .real [1,1] "tmp3"=some (fun _ => mean))
    (hr : s.regs .real [1,1] "tmp10"=some (fun _ => inverse))
    (hx : s.regs .nat [1,1] "x0"=some (fun _ => row))
    (bounds : RegionName → Nat)
    (bx : ∀ j,j<D → j+D*row<bounds X)
    (bw : ∀ j,j<D → j<bounds W) (bb : ∀ j,j<D → j<bounds Bias)
    (byOut : ∀ j,j<D → j+D*row<bounds Y) :
    Safety.loopSafe (withEquality M eq32) bounds "k" start (start+count)
      (normalizeBody reuse X W Bias Y S D B rt) s := by
  induction count generalizing s start with
  | zero => simp [Safety.loopSafe_unfold]
  | succ count ih =>
    obtain ⟨u,hu,_,hreg⟩ := normalize_chunk_run M eq32 s X W Bias Y S D B start row rt reuse mean inverse hm hr hx
    rw [Safety.loopSafe_unfold,if_pos (by omega)]
    refine ⟨normalize_safe M eq32 s X W Bias Y S D B start row rt reuse mean inverse hm hr hx bounds bx bw bb byOut,?_⟩
    rw [hu]
    simpa only [show start+Nat.succ count=(start+1)+count by omega] using ih u (start+1) hreg.mean hreg.rstd hreg.row

theorem source_safe (M : Algebra α) (eq32 : α → α → Bool) (reuse : Bool)
    (s : State α) (X W Bias Y Mean Rstd : RegionName) (S D B : Nat) (rt : ReductionTree B)
    (sep : X ≠ Mean) (bounds : RegionName → Nat)
    (bx : ∀ j,j<D → j+D*s.pids 0<bounds X)
    (bw : ∀ j,j<D → j<bounds W) (bb : ∀ j,j<D → j<bounds Bias)
    (byOut : ∀ j,j<D → j+D*s.pids 0<bounds Y)
    (bm : s.pids 0<bounds Mean) (br : s.pids 0<bounds Rstd) :
    TraceSafe (withEquality M eq32) bounds
      (SourceCode.selected reuse X W Bias Y Mean Rstd S D B (tree rt)) s := by
  obtain ⟨t,ran,ready⟩ := stats_run (withEquality M eq32) s X Mean Rstd S D B rt sep
  unfold TraceSafe
  rw [source_split]
  apply append_safe _ _ _ _ _ _ (stats_safe (withEquality M eq32) s X Mean Rstd S D B rt bounds bx bm br) ran
  apply Safety.traceSafeList_cons_intro
  · rw [Safety.stepSafe_unfold]
    simpa only [Nat.zero_add] using normalize_loop_safe M eq32 t X W Bias Y S D B 0 (blockCount D B) (s.pids 0) rt reuse _ _
      ready.mean ready.inverse ready.row bounds bx bw bb byOut
  · intro u _; exact Safety.traceSafeList_nil _ _ _

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.LayerNormWelfold.Readable
