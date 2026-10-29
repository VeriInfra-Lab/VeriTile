import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.RawAccumulate
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Statistics
import bench.repairs.tritonbench_g.layer_norm_welfold.IO

namespace VeriTile.Bench.Repairs.LayerNormWelfold.Raw
open VeriTile.Triton FP.Structural Reuse
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

structure Summed (phase : Bool) (s t : State α) (B : Nat) (mean total : α) : Prop where
  mem : t.mem=s.mem
  pids : t.pids=s.pids
  undef : t.undef=s.undef
  row : t.regs .nat [1,1] "x0"=some (fun _ => s.pids 0)
  base : t.regs .nat [1,B] "rbase"=some (fun i => i.2.1.val)
  total : t.regs .real [1,1] (if phase then "tmp12" else "tmp3")=some (fun _ => total)
  mean : phase=Bool.true → t.regs .real [1,1] "tmp6"=some (fun _ => mean)

structure MeanStored (s t : State α) (Mean : RegionName) (B : Nat) (mean : α) : Prop where
  mem : t.mem=(s.write Mean (s.pids 0) (.mk .real mean)).mem
  pids : t.pids=s.pids
  undef : t.undef=s.undef
  row : t.regs .nat [1,1] "x0"=some (fun _ => s.pids 0)
  base : t.regs .nat [1,B] "rbase"=some (fun i => i.2.1.val)
  mean : t.regs .real [1,1] "tmp6"=some (fun _ => mean)

structure InverseStored (s t : State α) (Rstd : RegionName) (B : Nat)
    (mean inverse : α) : Prop where
  mem : t.mem=(s.write Rstd (s.pids 0) (.mk .real inverse)).mem
  pids : t.pids=s.pids
  undef : t.undef=s.undef
  row : t.regs .nat [1,1] "x0"=some (fun _ => s.pids 0)
  base : t.regs .nat [1,B] "rbase"=some (fun i => i.2.1.val)
  mean : t.regs .real [1,1] "tmp6"=some (fun _ => mean)
  inverse : t.regs .real [1,1] "tmp18"=some (fun _ => inverse)

theorem tree_eval {B : Nat} (M : Algebra α) (s : State α) (name : RegName)
    (values : Fin B → α) (h : s.regs .real [1,B] name=some (fun i => values i.2.1))
    (tree : ReductionTree B) :
    evalOp M (some .fp32)
      (DSL.WelfordSource.sumExpression (.ref .real [1,B] name) (Readable.tree tree)) s =
      some (fun _ => treeValue M values tree) := by
  induction tree with
  | leaf i => simp [Readable.tree,DSL.WelfordSource.sumExpression,treeValue,evalOp_unfold,h]
  | node l r hl hr =>
      simp only [Readable.tree,DSL.WelfordSource.sumExpression,treeValue]
      rw [evalOp_unfold]
      simp [hl,hr,numeric,FP.Structural.bop]
      rfl

theorem sum_run (M : Algebra α) (phase : Bool) (s t : State α) (S D B : Nat)
    (tree : ReductionTree B) (values : Fin B → α) (mean : α)
    (h : Ready phase s t values mean) :
    ∃ u,run M (sumCode phase S D B (Readable.tree tree)) t=some u ∧
      Summed phase s u B mean (treeValue M values tree) := by
  have he := tree_eval M t (if phase then "_tmp12" else "_tmp3") values h.values tree
  cases phase <;> simp only [Bool.false_eq_true,↓reduceIte] at he
  all_goals
    simp [sumCode,code,SourceCode.rowKernel,ComputeKernel.surfaceBody,run,step,
      evalExpr,evalComputeOp,Option.bind,ComputeDType.eraseDType,
      ofFloat,toFloat]
  all_goals
    simp only [evalOp_unfold,he,Option.bind_some]
  all_goals try rfl
  all_goals constructor <;> simp [State.setReg,h.mem,h.pids,h.undef,h.row,h.base,h.mean,Fin.eq_zero]

theorem mean_run (M : Algebra α) (s t : State α) (Mean : RegionName) (S D B : Nat)
    (tree : DSL.WelfordSource.Tree B) (mean total : α) (h : Summed Bool.false s t B mean total) :
    ∃ u,run M (meanTail Mean S D B tree) t=some u ∧
      MeanStored s u Mean B (M.binary (some .fp32) .real .div total (M.fromNat (some .fp32) D)) := by
  have htotal := h.total
  dsimp at htotal
  simp [meanTail,code,SourceCode.rowKernel,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
    numeric,FP.Structural.bop,Region.cast,ofFloat,toFloat,htotal,h.row,
    store,TileShape.allIndices,List.finRange_succ,Fin.eq_zero]
  constructor <;> simp [State.setReg,State.write,h.mem,h.pids,h.undef,h.row,h.base,Fin.eq_zero]
  all_goals rfl

theorem reset_mean_run (M : Algebra α) (s t : State α) (Mean : RegionName)
    (S D B : Nat) (tree : DSL.WelfordSource.Tree B) (mean : α)
    (h : MeanStored s t Mean B mean) :
    ∃ u,run M (reset S D B tree) t=some u ∧
      Ready Bool.true (s.write Mean (s.pids 0) (.mk .real mean)) u (fun _ : Fin B => zero M) mean := by
  simp [reset,code,SourceCode.rowKernel,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,ofFloat,toFloat]
  constructor <;> simp [State.setReg,State.write,zero,h.mem,h.pids,h.undef,h.row,h.base,h.mean]

theorem inverse_run (M : Algebra α) (s t : State α) (Rstd : RegionName) (S D B : Nat)
    (tree : DSL.WelfordSource.Tree B) (mean total : α) (h : Summed Bool.true s t B mean total) :
    ∃ u,run M (inverseTail Rstd S D B tree) t=some u ∧
      InverseStored s u Rstd B mean (inverseFromTotal M D total) := by
  have htotal := h.total
  dsimp at htotal
  simp [inverseTail,code,SourceCode.rowKernel,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
    numeric,FP.Structural.bop,Region.cast,ofFloat,toFloat,htotal,h.row,
    store,TileShape.allIndices,List.finRange_succ,Fin.eq_zero]
  constructor <;> simp [State.setReg,State.write,inverseFromTotal,
    h.mem,h.pids,h.undef,h.row,h.base,h.mean,Fin.eq_zero]
  all_goals rfl

structure StatisticsReady (s t : State α) (Mean Rstd : RegionName) (B : Nat)
    (mean inverse : α) : Prop where
  mem : t.mem=((s.write Mean (s.pids 0) (.mk .real mean)).write Rstd (s.pids 0) (.mk .real inverse)).mem
  pids : t.pids=s.pids
  undef : t.undef=s.undef
  row : t.regs .nat [1,1] "x0"=some (fun _ => s.pids 0)
  base : t.regs .nat [1,B] "rbase"=some (fun i => i.2.1.val)
  mean : t.regs .real [1,1] "tmp6"=some (fun _ => mean)
  inverse : t.regs .real [1,1] "tmp18"=some (fun _ => inverse)

theorem stats_run (M : Algebra α) (s : State α)
    (X Mean Rstd : RegionName) (S D B : Nat) (rt : ReductionTree B) (hB : 0<B) (hX : X≠Mean) :
    ∃ t,run M (stats X Mean Rstd S D B (Readable.tree rt)) s=some t ∧
      StatisticsReady s t Mean Rstd B (meanValue M s X D B rt) (inverseValue M s X D B rt) := by
  obtain ⟨a,ha,hia⟩ := prefix_run M s S D B (Readable.tree rt)
  obtain ⟨b,hb,hib⟩ := phase_loop M Bool.false s a X S D B 0 (blockCount D B) (Readable.tree rt) hB (by simp) _ _ hia
  obtain ⟨c,hc,hic⟩ := sum_run M Bool.false s b S D B rt _ _ hib
  obtain ⟨d,hd,hid⟩ := mean_run M s c Mean S D B (Readable.tree rt) _ _ hic
  obtain ⟨e,he,hie⟩ := reset_mean_run M s d Mean S D B (Readable.tree rt) _ hid
  obtain ⟨f,hf,hif⟩ := phase_loop M Bool.true _ e X S D B 0 (blockCount D B) (Readable.tree rt) hB (by simp) _ _ hie
  obtain ⟨g,hg,hig⟩ := sum_run M Bool.true _ f S D B rt _ _ hif
  obtain ⟨t,ht,hit⟩ := inverse_run M _ g Rstd S D B (Readable.tree rt) _ _ hig
  have hxmem : (s.write Mean (s.pids 0) (.mk .real (meanValue M s X D B rt))).mem X=s.mem X := by
    funext off
    exact State.write_other _ _ _ _ _ _ (.inl hX)
  have hv := recurrence_congr M Bool.true
    (s.write Mean (s.pids 0) (.mk .real (meanValue M s X D B rt))) s X D B
    0 (blockCount D B) (meanValue M s X D B rt) (fun _ => zero M) hxmem rfl
  simp only [Nat.zero_mul] at hb hf
  have hbr : run M [.forRange "roffset" 0 D B (phaseBody Bool.false X S D B (Readable.tree rt))] a=some b := by
    simp [run,step,hb]
  have hfr : run M [.forRange "roffset" 0 D B (phaseBody Bool.true X S D B (Readable.tree rt))] e=some f := by
    simp [run,step,hf]
  refine ⟨t,?_,?_⟩
  · rw [stats_split]
    simp only [run_append,ha,hbr,hc,hd,he,hfr,hg,ht,Option.bind]
  · change InverseStored (s.write Mean (s.pids 0) (.mk .real (meanValue M s X D B rt))) t Rstd B
      (meanValue M s X D B rt)
      (inverseFromTotal M D (treeValue M
        (recurrence M Bool.true (s.write Mean (s.pids 0) (.mk .real (meanValue M s X D B rt)))
          X D B (meanValue M s X D B rt) 0 (blockCount D B) (fun _ => zero M)) rt)) at hit
    rw [hv] at hit
    exact ⟨hit.mem,hit.pids,hit.undef,hit.row,hit.base,hit.mean,hit.inverse⟩
#axiomsClean stats_run
#axiomsClean tree_eval
#axiomsClean sum_run
#axiomsClean mean_run
#axiomsClean reset_mean_run
#axiomsClean inverse_run
end
end VeriTile.Bench.Repairs.LayerNormWelfold.Raw
