import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Accumulate
namespace VeriTile.Bench.Repairs.LayerNormWelfold
open VeriTile.Triton FP.Structural Reuse
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def allLanes {α : Type} [Inhabited α] (M : Algebra α) (phase : Bool) (s : State α)
    (X : RegionName) (D B : Nat) (mean : α) : Fin B → α :=
  recurrence M phase s X D B mean 0 (blockCount D B) (fun _ => zero M)

def meanValue {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (X : RegionName) (D B : Nat) (tree : ReductionTree B) : α :=
  M.binary (some .fp32) .real .div
    (treeValue M (allLanes M Bool.false s X D B (zero M)) tree)
    (M.fromNat (some .fp32) D)

def inverseValue {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (X : RegionName) (D B : Nat) (tree : ReductionTree B) : α :=
  M.unary (some .fp32) .libdeviceRsqrt (M.binary (some .fp32) .real .add
    (M.binary (some .fp32) .real .div
      (treeValue M (allLanes M Bool.true s X D B (meanValue M s X D B tree)) tree)
      (M.fromNat (some .fp32) D)) (M.literal (some .fp32) .real (1e-05)))

theorem init_run {α : Type} [Inhabited α] (M : Algebra α) (s : State α) (S B : Nat) :
    ∃ t,run M (init S B) s=some t ∧ Ready s t (fun _ : Fin B => zero M) (zero M) := by
  simp [init,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,natLt,numericLt,
    Region.cast,ofFloat,toFloat]
  constructor <;> simp [State.setReg,zero,Fin.eq_zero]
  funext i
  simp [FP.Structural.bop]

theorem reset_run {α : Type} [Inhabited α] (M : Algebra α) (s t : State α)
    {B : Nat} (values : Fin B → α) (mean : α) (h : Ready s t values mean) :
    ∃ u,run M (reset B) t=some u ∧ Ready s u (fun _ : Fin B => zero M) mean := by
  simp [reset,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,ofFloat,toFloat]
  constructor <;> simp [State.setReg,zero,h.mem,h.pids,h.undef,h.row,h.mean]

structure Summed {α : Type} (s t : State α) (mean total : α) : Prop where
  mem : t.mem=s.mem
  pids : t.pids=s.pids
  undef : t.undef=s.undef
  row : t.regs .nat [1,1] "x0"=some (fun _ => s.pids 0)
  mean : t.regs .real [1,1] "tmp3"=some (fun _ => mean)
  total : t.regs .real [1,1] "total"=some (fun _ => total)

theorem sum_run {α : Type} [Inhabited α] {B : Nat} (M : Algebra α) (s t : State α)
    (values : Fin B → α) (mean : α) (h : Ready s t values mean) (tree : ReductionTree B) :
    ∃ u,run M (sumCode tree) t=some u ∧ Summed s u mean (treeValue M values tree) := by
  have he := sum_expr_eval M t values h.values tree
  have hf : evalOp M (some .fp32) (.full [1,1] (sumExpr tree)) t=some (fun _ => treeValue M values tree) := by
    rw [evalOp_unfold]
    simp only [he,Option.bind]
    rfl
  refine ⟨t.setReg "total" .real [1,1] (fun _ => treeValue M values tree),?_,?_⟩
  · simp only [sumCode,run,step,evalExpr,evalComputeOp,ComputeDType.eraseDType]
    rw [hf]
    rfl
  · constructor <;> simp [State.setReg,h.mem,h.pids,h.undef,h.row,h.mean]

structure MeanStored {α : Type} (s t : State α) (Mean : RegionName) (value : α) : Prop where
  mem : t.mem=(s.write Mean (s.pids 0) (.mk .real value)).mem
  pids : t.pids=s.pids
  undef : t.undef=s.undef
  row : t.regs .nat [1,1] "x0"=some (fun _ => s.pids 0)
  mean : t.regs .real [1,1] "tmp3"=some (fun _ => value)

theorem mean_run {α : Type} [Inhabited α] (M : Algebra α) (s t : State α)
    (Mean : RegionName) (D : Nat) (mean total : α) (h : Summed s t mean total) :
    ∃ u,run M (meanCode Mean D) t=some u ∧
      MeanStored s u Mean (M.binary (some .fp32) .real .div total (M.fromNat (some .fp32) D)) := by
  simp [meanCode,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,Region.cast,
    ofFloat,toFloat,h.total,h.row,store,TileShape.allIndices,List.finRange_succ,Fin.eq_zero]
  constructor <;> simp [State.setReg,State.write,h.mem,h.pids,h.undef,h.row,Fin.eq_zero]
  all_goals rfl

structure StatisticsReady {α : Type} (s t : State α) (Mean Rstd : RegionName)
    (mean inverse : α) : Prop where
  mem : t.mem=((s.write Mean (s.pids 0) (.mk .real mean)).write Rstd (s.pids 0) (.mk .real inverse)).mem
  pids : t.pids=s.pids
  undef : t.undef=s.undef
  row : t.regs .nat [1,1] "x0"=some (fun _ => s.pids 0)
  mean : t.regs .real [1,1] "tmp3"=some (fun _ => mean)
  inverse : t.regs .real [1,1] "tmp10"=some (fun _ => inverse)

structure InverseStored {α : Type} (s t : State α) (Rstd : RegionName)
    (mean inverse : α) : Prop where
  mem : t.mem=(s.write Rstd (s.pids 0) (.mk .real inverse)).mem
  pids : t.pids=s.pids
  undef : t.undef=s.undef
  row : t.regs .nat [1,1] "x0"=some (fun _ => s.pids 0)
  mean : t.regs .real [1,1] "tmp3"=some (fun _ => mean)
  inverse : t.regs .real [1,1] "tmp10"=some (fun _ => inverse)

def inverseFromTotal {α : Type} (M : Algebra α) (D : Nat) (total : α) : α :=
  M.unary (some .fp32) .libdeviceRsqrt (M.binary (some .fp32) .real .add
    (M.binary (some .fp32) .real .div total (M.fromNat (some .fp32) D))
    (M.literal (some .fp32) .real (1e-05)))

theorem inverse_run {α : Type} [Inhabited α] (M : Algebra α) (s t : State α)
    (Rstd : RegionName) (D : Nat) (mean total : α) (h : Summed s t mean total) :
    ∃ u,run M (inverseCode Rstd D) t=some u ∧
      InverseStored s u Rstd mean (inverseFromTotal M D total) := by
  simp [inverseCode,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,Region.cast,
    ofFloat,toFloat,h.total,h.row,store,TileShape.allIndices,List.finRange_succ,Fin.eq_zero]
  constructor <;> simp [State.setReg,State.write,inverseFromTotal,h.mem,h.pids,h.undef,h.row,h.mean,Fin.eq_zero]
  all_goals rfl

theorem reset_mean_run {α : Type} [Inhabited α] (M : Algebra α) (s t : State α)
    (Mean : RegionName) (B : Nat) (mean : α) (h : MeanStored s t Mean mean) :
    ∃ u,run M (reset B) t=some u ∧
      Ready (s.write Mean (s.pids 0) (.mk .real mean)) u (fun _ : Fin B => zero M) mean := by
  simp [reset,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,ofFloat,toFloat]
  constructor <;> simp [State.setReg,State.write,zero,h.mem,h.pids,h.undef,h.row,h.mean]

theorem recurrence_congr {α : Type} [Inhabited α] (M : Algebra α) (phase : Bool)
    (s t : State α) (X : RegionName) (D B start count : Nat) (mean : α) (previous : Fin B → α)
    (hm : s.mem X=t.mem X) (hp : s.pids 0=t.pids 0) :
    recurrence M phase s X D B mean start count previous=
      recurrence M phase t X D B mean start count previous := by
  induction count generalizing start previous with
  | zero => rfl
  | succ count ih =>
    simp only [recurrence]
    rw [ih]
    congr 1
    funext i
    simp only [advance,loaded,hm,hp]

theorem whole_statistics {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (X Mean Rstd : RegionName) (S D B : Nat) (tree : ReductionTree B) (hX : X≠Mean) :
    ∃ t,FP.Structural.exec M (statisticsKernel X Mean Rstd S D B tree) s=some t ∧
      StatisticsReady s t Mean Rstd (meanValue M s X D B tree) (inverseValue M s X D B tree) := by
  obtain ⟨a,ha,hia⟩ := init_run M s S B
  obtain ⟨b,hb,hib⟩ := chunks_run M Bool.false s a X D B 0 (blockCount D B) _ _ hia
  obtain ⟨c,hc,hic⟩ := sum_run M s b _ _ hib tree
  obtain ⟨d,hd,hid⟩ := mean_run M s c Mean D _ _ hic
  obtain ⟨e,he,hie⟩ := reset_mean_run M s d Mean B _ hid
  obtain ⟨f,hf,hif⟩ := chunks_run M Bool.true _ e X D B 0 (blockCount D B) _ _ hie
  obtain ⟨g,hg,hig⟩ := sum_run M _ f _ _ hif tree
  obtain ⟨t,ht,hit⟩ := inverse_run M _ g Rstd D _ _ hig
  have hxmem : (s.write Mean (s.pids 0) (.mk .real (meanValue M s X D B tree))).mem X=s.mem X := by
    funext off
    exact State.write_other _ _ _ _ _ _ (.inl hX)
  have hv := recurrence_congr M Bool.true
    (s.write Mean (s.pids 0) (.mk .real (meanValue M s X D B tree))) s X D B
    0 (blockCount D B) (meanValue M s X D B tree) (fun _ => zero M) hxmem rfl
  refine ⟨t,?_,?_⟩
  · simp only [FP.Structural.exec,statisticsKernel,ComputeKernel.surfaceBody,run_append,
      ha,hb,hc,hd,he,hf,hg,ht,Option.bind]
  · change InverseStored (s.write Mean (s.pids 0) (.mk .real (meanValue M s X D B tree))) t Rstd
      (meanValue M s X D B tree)
      (inverseFromTotal M D (treeValue M
        (recurrence M Bool.true (s.write Mean (s.pids 0) (.mk .real (meanValue M s X D B tree)))
          X D B (meanValue M s X D B tree) 0 (blockCount D B) (fun _ => zero M)) tree)) at hit
    rw [hv] at hit
    exact ⟨hit.mem,hit.pids,hit.undef,hit.row,hit.mean,hit.inverse⟩

#axiomsClean init_run
#axiomsClean reset_run
#axiomsClean sum_run
#axiomsClean mean_run
#axiomsClean inverse_run
#axiomsClean whole_statistics
end
end VeriTile.Bench.Repairs.LayerNormWelfold
