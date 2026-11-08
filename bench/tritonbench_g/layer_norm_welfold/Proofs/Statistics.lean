import bench.tritonbench_g.layer_norm_welfold.Proofs.Accumulate
import bench.tritonbench_g.layer_norm_welfold.Proofs.ReadableSyntax
namespace VeriTile.Bench.Repairs.LayerNormWelfold
open VeriTile.Triton FP.Structural Reuse
noncomputable section
set_option maxHeartbeats 1000000
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
      (M.fromNat (some .fp32) D)) (M.literal none .real (1e-05)))

theorem init_run {α : Type} [Inhabited α] (M : Algebra α) (s : State α) (S B : Nat) :
    ∃ t,run M (init S B) s=some t ∧ Ready Bool.false s t (fun _ : Fin B => zero M) (zero M) := by
  simp [init,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,natLt,numericLt,
    Region.cast,ofFloat,toFloat]
  constructor <;> simp [State.setReg,zero,Fin.eq_zero]
  funext i
  simp [FP.Structural.bop]
  rfl

theorem reset_run {α : Type} [Inhabited α] (M : Algebra α) (s t : State α)
    {B : Nat} (values : Fin B → α) (mean : α) (h : Ready Bool.true s t values mean) :
    ∃ u,run M (reset B) t=some u ∧ Ready Bool.true s u (fun _ : Fin B => zero M) mean := by
  simp [reset,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,ofFloat,toFloat]
  constructor <;> simp [State.setReg,zero,h.mem,h.pids,h.undef,h.row,h.rbase,h.mean]

structure Summed (phase : Bool) {α : Type} (s t : State α) (B : Nat) (mean total : α) : Prop where
  mem : t.mem=s.mem
  pids : t.pids=s.pids
  undef : t.undef=s.undef
  row : t.regs .nat [1,1] "x0"=some (fun _ => s.pids 0)
  rbase : t.regs .nat [1,B] "rbase"=some (fun i => i.2.1.val)
  mean : phase = Bool.true → t.regs .real [1,1] "tmp6"=some (fun _ => mean)
  total : t.regs .real [1,1] (if phase then "tmp12" else "tmp3")=some (fun _ => total)

private theorem ordered_view_eval {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (op : Op .real []) (value : α)
    (h : evalOp M (some .fp32) op s = some (fun _ => value)) :
    evalOp M (some .fp32)
      (.expandDim (shape := [1]) ⟨1,by decide⟩ (.reshape [1] (.full [1,1] op))) s =
        some (fun _ => value) := by
  have filled : evalOp M (some .fp32) (.full [1,1] op) s = some (fun _ => value) := by
    rw [evalOp_unfold]
    simp only [h,Option.bind_some]
    rfl
  have reshaped : evalOp M (some .fp32) (.reshape [1] (.full [1,1] op)) s =
      some (fun _ => value) := by
    rw [evalOp_unfold]
    simp only [filled,Option.bind_some]
    rfl
  rw [evalOp_unfold]
  simp only [reshaped,Option.bind_some]
  rfl

theorem sum_run {α : Type} [Inhabited α] {B : Nat} {phase : Bool} (M : Algebra α) (s t : State α)
    (values : Fin B → α) (mean : α) (h : Ready phase s t values mean) (tree : ReductionTree B) :
    ∃ u,run M (sumCode phase tree) t=some u ∧ Summed phase s u B mean (treeValue M values tree) := by
  have he := sum_expr_eval M t values phase h.values tree
  rw [←Readable.sum_expression phase tree] at he
  dsimp only [Readable.tree] at he
  have view := ordered_view_eval M t _ _ he
  have stored := congrArg (fun result : Option (TileIndex [1,1] → α) =>
    result.bind (fun value => some (t.setReg (if phase then "tmp12" else "tmp3")
      .real [1,1] value))) view
  refine ⟨t.setReg (if phase then "tmp12" else "tmp3") .real [1,1]
    (fun _ => treeValue M values tree),?_,?_⟩
  · cases phase
    all_goals
      simpa only [sumCode,run,step,evalExpr,evalComputeOp,ComputeDType.eraseDType,
        Bool.false_eq_true,↓reduceIte,bind,pure,Option.bind_some,Option.bind_fun_some] using stored
  · cases phase <;> constructor <;>
      simp [State.setReg,h.mem,h.pids,h.undef,h.row,h.rbase,h.mean]

structure MeanStored {α : Type} (s t : State α) (Mean : RegionName) (B : Nat) (value : α) : Prop where
  mem : t.mem=(s.write Mean (s.pids 0) (.mk .real value)).mem
  pids : t.pids=s.pids
  undef : t.undef=s.undef
  row : t.regs .nat [1,1] "x0"=some (fun _ => s.pids 0)
  rbase : t.regs .nat [1,B] "rbase"=some (fun i => i.2.1.val)
  mean : t.regs .real [1,1] "tmp6"=some (fun _ => value)

theorem mean_run {α : Type} [Inhabited α] (M : Algebra α) (s t : State α)
    (Mean : RegionName) (D B : Nat) (mean total : α) (h : Summed Bool.false s t B mean total) :
    ∃ u,run M (meanCode Mean D) t=some u ∧
      MeanStored s u Mean B (M.binary (some .fp32) .real .div total (M.fromNat (some .fp32) D)) := by
  have hv := h.total
  simp at hv
  simp [meanCode,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,Region.cast,
    ofFloat,toFloat,hv,h.row,store,TileShape.allIndices,List.finRange_succ,Fin.eq_zero]
  constructor <;> simp [State.setReg,State.write,h.mem,h.pids,h.undef,h.row,h.rbase,Fin.eq_zero]
  all_goals rfl

structure StatisticsReady {α : Type} (s t : State α) (Mean Rstd : RegionName) (B : Nat)
    (mean inverse : α) : Prop where
  mem : t.mem=((s.write Mean (s.pids 0) (.mk .real mean)).write Rstd (s.pids 0) (.mk .real inverse)).mem
  pids : t.pids=s.pids
  undef : t.undef=s.undef
  row : t.regs .nat [1,1] "x0"=some (fun _ => s.pids 0)
  rbase : t.regs .nat [1,B] "rbase"=some (fun i => i.2.1.val)
  mean : t.regs .real [1,1] "tmp6"=some (fun _ => mean)
  inverse : t.regs .real [1,1] "tmp18"=some (fun _ => inverse)

structure InverseStored {α : Type} (s t : State α) (Rstd : RegionName) (B : Nat)
    (mean inverse : α) : Prop where
  mem : t.mem=(s.write Rstd (s.pids 0) (.mk .real inverse)).mem
  pids : t.pids=s.pids
  undef : t.undef=s.undef
  row : t.regs .nat [1,1] "x0"=some (fun _ => s.pids 0)
  rbase : t.regs .nat [1,B] "rbase"=some (fun i => i.2.1.val)
  mean : t.regs .real [1,1] "tmp6"=some (fun _ => mean)
  inverse : t.regs .real [1,1] "tmp18"=some (fun _ => inverse)

def inverseFromTotal {α : Type} (M : Algebra α) (D : Nat) (total : α) : α :=
  M.unary (some .fp32) .libdeviceRsqrt (M.binary (some .fp32) .real .add
    (M.binary (some .fp32) .real .div total (M.fromNat (some .fp32) D))
    (M.literal none .real (1e-05)))

theorem inverse_run {α : Type} [Inhabited α] (M : Algebra α) (s t : State α)
    (Rstd : RegionName) (D B : Nat) (mean total : α) (h : Summed Bool.true s t B mean total) :
    ∃ u,run M (inverseCode Rstd D) t=some u ∧
      InverseStored s u Rstd B mean (inverseFromTotal M D total) := by
  have hv := h.total
  have hm := h.mean
  simp at hv hm
  simp [inverseCode,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,Region.cast,
    ofFloat,toFloat,hv,h.row,store,TileShape.allIndices,List.finRange_succ,Fin.eq_zero]
  constructor <;> simp [State.setReg,State.write,inverseFromTotal,h.mem,h.pids,h.undef,h.row,h.rbase,hm,Fin.eq_zero]
  all_goals rfl

theorem reset_mean_run {α : Type} [Inhabited α] (M : Algebra α) (s t : State α)
    (Mean : RegionName) (B : Nat) (mean : α) (h : MeanStored s t Mean B mean) :
    ∃ u,run M (reset B) t=some u ∧
      Ready Bool.true (s.write Mean (s.pids 0) (.mk .real mean)) u (fun _ : Fin B => zero M) mean := by
  simp [reset,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,ofFloat,toFloat]
  constructor <;> simp [State.setReg,State.write,zero,h.mem,h.pids,h.undef,h.row,h.rbase,h.mean]

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

#axiomsClean init_run
#axiomsClean reset_run
#axiomsClean sum_run
#axiomsClean mean_run
#axiomsClean inverse_run
end
end VeriTile.Bench.Repairs.LayerNormWelfold
