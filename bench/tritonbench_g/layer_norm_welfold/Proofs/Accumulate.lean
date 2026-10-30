import bench.tritonbench_g.layer_norm_welfold.Proofs.Source
namespace VeriTile.Bench.Repairs.LayerNormWelfold
open VeriTile.Triton FP.Structural Reuse
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def zero {α : Type} (M : Algebra α) : α := M.literal (some .fp32) .real 0

def loaded {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (X : RegionName) (D B k : Nat) (i : Fin B) : α :=
  M.cast (some .fp32) .real .real (M.cast (some .fp32) .real .real
    (if k*B+i.val<D then (s.mem X (k*B+i.val+D*s.pids 0)).read .real else zero M))

def advance {α : Type} [Inhabited α] (M : Algebra α) (phase : Bool) (s : State α)
    (X : RegionName) (D B k : Nat) (mean : α) (previous : Fin B → α) : Fin B → α := fun i =>
  let value := loaded M s X D B k i
  let delta := M.binary (some .fp32) .real .sub value mean
  let contribution := if phase then
    if k*B+i.val<D then M.binary (some .fp32) .real .mul delta delta else zero M
    else value
  M.binary (some .fp32) .real .add (previous i) contribution

def recurrence {α : Type} [Inhabited α] (M : Algebra α) (phase : Bool) (s : State α)
    (X : RegionName) (D B : Nat) (mean : α) : Nat → Nat → (Fin B → α) → (Fin B → α)
  | _,0,previous => previous
  | start,count+1,previous => recurrence M phase s X D B mean (start+1) count
      (advance M phase s X D B start mean previous)

structure Ready {α : Type} (s t : State α) {B : Nat} (values : Fin B → α) (mean : α) : Prop where
  mem : t.mem=s.mem
  pids : t.pids=s.pids
  undef : t.undef=s.undef
  row : t.regs .nat [1,1] "x0" = some (fun _ => s.pids 0)
  values : t.regs .real [1,B] "acc" = some (fun i => values i.2.1)
  mean : t.regs .real [1,1] "tmp3" = some (fun _ => mean)

theorem chunk_run {α : Type} [Inhabited α] (M : Algebra α) (phase : Bool)
    (s t : State α) (X : RegionName) (D B k : Nat) (values : Fin B → α) (mean : α)
    (h : Ready s t values mean) :
    ∃ final,run M (chunk phase X D B k) t=some final ∧
      Ready s final (advance M phase s X D B k mean values) mean := by
  cases phase <;>
    simp [chunk,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,Region.cast,
      ofFloat,toFloat,h.values,h.mean,h.row,h.mem]
  all_goals constructor <;> simp [State.setReg,advance,loaded,zero,bop,ofFloat,toFloat,
    Fin.eq_zero,h.mem,h.pids,h.undef,h.row,h.mean]
  all_goals first | rfl | (funext i; first | (split <;> simp_all [Fin.eq_zero]) | simp [bop])
  all_goals try dsimp [FP.Structural.bop]
  all_goals try simp_all
  all_goals norm_num [FP.Structural.bop]
  all_goals rename_i i hi
  all_goals simp only [if_neg (Nat.not_lt_of_ge hi)]

theorem chunks_run {α : Type} [Inhabited α] (M : Algebra α) (phase : Bool)
    (s t : State α) (X : RegionName) (D B start count : Nat) (values : Fin B → α) (mean : α)
    (h : Ready s t values mean) :
    ∃ final,run M (chunks phase X D B start count) t=some final ∧
      Ready s final (recurrence M phase s X D B mean start count values) mean := by
  induction count generalizing start t values with
  | zero => exact ⟨t,by simp [chunks,run],h⟩
  | succ count ih =>
    obtain ⟨u,hu,hp⟩ := chunk_run M phase s t X D B start values mean h
    obtain ⟨v,hv,hq⟩ := ih u (start+1) _ hp
    exact ⟨v,by simp [chunks,run_append,hu,hv],hq⟩

def treeValue {α : Type} {B : Nat} (M : Algebra α) (values : Fin B → α) : ReductionTree B → α
  | .leaf i => values i
  | .node l r => M.binary (some .fp32) .real .add (treeValue M values l) (treeValue M values r)

theorem sum_expr_eval {α : Type} [Inhabited α] {B : Nat} (M : Algebra α)
    (s : State α) (values : Fin B → α)
    (h : s.regs .real [1,B] "acc" = some (fun i => values i.2.1)) (tree : ReductionTree B) :
    evalOp M (some .fp32) (sumExpr tree) s = some (fun _ => treeValue M values tree) := by
  induction tree with
  | leaf i => simp [sumExpr,treeValue,evalOp_unfold,h]
  | node l r il ir =>
    rw [sumExpr,evalOp_unfold]
    simp [il,ir,treeValue,numeric,FP.Structural.bop]
    rfl

#axiomsClean chunks_run
#axiomsClean sum_expr_eval
end
end VeriTile.Bench.Repairs.LayerNormWelfold
