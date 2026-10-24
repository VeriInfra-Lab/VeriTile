import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.Model

/-! Independent logical formulas are reused from the existing LayerNorm math
library. The ordered graph has additional zero leaves and a different DX
parenthesization; the proofs below account for those differences explicitly. -/
namespace VeriTile.Bench.Repairs.LayerNormTriton.RealProof
open VeriTile.Triton
open VeriTile.Bench.Repairs.FastLayernorm (MulAdd)
noncomputable section
namespace Spec
abbrev mean := VeriTile.Bench.Repairs.FastLayernorm.RealProof.meanSpec
abbrev variance := VeriTile.Bench.Repairs.FastLayernorm.RealProof.varianceSpec
abbrev inverse := VeriTile.Bench.Repairs.FastLayernorm.RealProof.invSpec
abbrev output := VeriTile.Bench.Repairs.FastLayernorm.RealProof.outputSpec
abbrev gradient := VeriTile.Bench.Repairs.FastLayernorm.RealProof.gradientSpec
abbrev weightGradient := VeriTile.Bench.Repairs.FastLayernorm.RealProof.weightGradientSpec
abbrev biasGradient := VeriTile.Bench.Repairs.FastLayernorm.RealProof.biasGradientSpec
end Spec
set_option maxRecDepth 24000
set_option maxHeartbeats 2000000

def ops : Ops ℝ := ⟨0,1,fun n => n,(·+·),(·-·),(·*·),(·/·),fun a b c => a*b+c,Real.sqrt⟩

theorem ordered_add (swap : Bool) (a b : ℝ) : ordered ops.add swap a b=a+b := by
  cases swap <;> simp [ordered,ops,add_comm]
theorem ordered_mul (swap : Bool) (a b : ℝ) : ordered ops.mul swap a b=a*b := by
  cases swap <;> simp [ordered,ops,mul_comm]
theorem mulAdd_real (kind : MulAdd) (a b c : ℝ) : mulAdd ops kind a b c=a*b+c := by
  cases kind with
  | separate sm sa => rw [mulAdd,ordered_add,ordered_mul]
  | fused sm => cases sm <;> simp [mulAdd,ops,mul_comm]

theorem sumTree_real (tree : SumTree B) (x : Fin B → ℝ) :
    tree.eval ops x=(tree.leaves.map x).sum := by
  induction tree with
  | zero => simp [SumTree.eval,SumTree.leaves,ops]
  | leaf j => simp [SumTree.eval,SumTree.leaves]
  | add a b ia ib => simp only [SumTree.eval,SumTree.leaves,ia,ib,List.map_append,List.sum_append]; rfl

theorem productTree_real (tree : ProductTree B) (x y : Fin B → ℝ) :
    tree.eval ops x y=(tree.leaves.map (fun j => x j*y j)).sum := by
  induction tree with
  | zero => simp [ProductTree.eval,ProductTree.leaves,ops]
  | product j swap => simp [ProductTree.eval,ProductTree.leaves,ordered_mul]
  | add a b ia ib => simp only [ProductTree.eval,ProductTree.leaves,ia,ib,List.map_append,List.sum_append]; rfl
  | fma j swap rest ih => cases swap <;> simp only [ProductTree.eval,Bool.false_eq_true,if_false,if_true,ih,ProductTree.leaves,List.map_cons,List.sum_cons] <;> simp [ops,mul_comm]

theorem divide_real (plan : Plan N B) (x : ℝ) : divideCount ops plan x=x/(N : ℝ) := by
  unfold divideCount
  split
  · rename_i h
    rw [plan.unitOnly h]
    simp
  · rfl

theorem sum_padded (tree : SumTree B) (complete : tree.leaves.Perm (List.finRange B))
    (covers : N≤B) (x : Nat → ℝ) :
    tree.eval ops (masked ops N x)=∑ j ∈ Finset.range N,x j := by
  rw [sumTree_real,(complete.map _).sum_eq,←List.ofFn_eq_map,List.sum_ofFn]
  change (∑ j : Fin B,if j.val<N then x j.val else 0)=_
  rw [Fin.sum_univ_eq_sum_range (fun j => if j<N then x j else 0)]
  exact VeriTile.Bench.TritonBenchG.LayerNormFwd.Formula.sum_padded x covers

theorem product_padded (tree : ProductTree B) (complete : tree.leaves.Perm (List.finRange B))
    (covers : N≤B) (x y : Nat → ℝ) :
    tree.eval ops (masked ops N x) (masked ops N y)=∑ j ∈ Finset.range N,x j*y j := by
  rw [productTree_real,(complete.map _).sum_eq,←List.ofFn_eq_map,List.sum_ofFn]
  have hp : (fun j : Fin B => masked ops N x j*masked ops N y j)=
      fun j : Fin B => if j.val<N then x j.val*y j.val else 0 := by
    funext j
    by_cases h : j.val<N <;> simp [masked,h,ops]
  rw [hp,Fin.sum_univ_eq_sum_range (fun j => if j<N then x j*y j else 0)]
  exact VeriTile.Bench.TritonBenchG.LayerNormFwd.Formula.sum_padded (fun j => x j*y j) covers

theorem mean_real (plan : Plan N B) (x : Nat → ℝ) (lane : Nat) : mean ops plan x lane=Spec.mean N x := by
  unfold mean
  rw [divide_real,sum_padded _ (plan.meanComplete lane) plan.covers]
  rfl

theorem centered_real (plan : Plan N B) (x : Nat → ℝ) :
    centered ops plan x=masked ops N (fun j => x j-Spec.mean N x) := by
  funext j
  simp only [centered,mean_real,masked]
  rfl

theorem variance_real (plan : Plan N B) (x : Nat → ℝ) (lane : Nat) :
    variance ops plan x lane=Spec.variance N x := by
  unfold variance
  rw [divide_real,centered_real,product_padded _ (plan.varianceComplete lane) plan.covers]
  simp only [Spec.variance,VeriTile.Bench.Repairs.FastLayernorm.RealProof.varianceSpec,pow_two]

theorem statistics_real (plan : Plan N B) (x : Nat → ℝ) (eps : ℝ) (lane : Nat) :
    statistics ops plan x eps lane=⟨Spec.mean N x,Spec.inverse N x eps⟩ := by
  simp only [statistics,mean_real,ordered_add,variance_real]
  rfl

theorem normed_real (plan : Plan N B) (stats : Reuse.Statistics ℝ) (x : ℝ) :
    normed ops plan stats x=(x-stats.mean)*stats.inverse := by
  unfold normed
  rw [ordered_mul]
  rfl

theorem forward_real (plan : Plan N B) (x w bias : Nat → ℝ) (eps : ℝ) (col : Nat) :
    forwardValue ops plan x w bias eps col=Spec.output N x w bias eps col := by
  simp only [forwardValue,statistics_real,mulAdd_real,normed_real]
  rfl

theorem weightedDy_real (plan : Plan N B) (dy w : Nat → ℝ) (col : Nat) :
    weightedDy ops plan dy w col=dy col*w col := by
  simp [weightedDy,ordered_mul,mul_comm]

theorem backward_real (plan : Plan N B) (stats : Reuse.Statistics ℝ)
    (x dy w : Nat → ℝ) (col : Nat) :
    backwardValue ops plan stats x dy w col=Spec.gradient N x dy w stats.mean stats.inverse col := by
  simp only [backwardValue,divide_real,mulAdd_real,ordered_mul,
    product_padded _ (plan.firstComplete col) plan.covers,
    sum_padded _ (plan.secondComplete col) plan.covers,normed_real,weightedDy_real]
  have hm : (∑ j ∈ Finset.range N,(x j-stats.mean)*stats.inverse*(dy j*w j))=
      ∑ j ∈ Finset.range N,dy j*w j*((x j-stats.mean)*stats.inverse) := by
    apply Finset.sum_congr rfl
    intro j _
    ring
  change (dy col*w col-(((x col-stats.mean)*stats.inverse)*(_/(N : ℝ))+_))*stats.inverse=_
  rw [hm]
  unfold Spec.gradient VeriTile.Bench.Repairs.FastLayernorm.RealProof.gradientSpec
  ring

/-- The first row has no add-zero edge; the following fold lemma handles both
branches without imposing any floating-point associativity law. -/
def optionValue (a : Option (ℝ×ℝ)) : ℝ×ℝ := a.getD (0,0)

theorem groupStep_real (plan : Plan N B) (x dy : Nat → Nat → ℝ)
    (stats : Nat → Reuse.Statistics ℝ) (column row : Nat) (a : Option (ℝ×ℝ)) :
    optionValue (groupStep ops plan x dy stats column row a)=
      ((optionValue a).1+dy row column*((x row column-(stats row).mean)*(stats row).inverse),
       (optionValue a).2+dy row column) := by
  cases a with
  | none => simp only [groupStep,optionValue,normed_real,ordered_mul,Option.getD_none,Option.getD_some]; simp [ops]
  | some previous => simp [groupStep,optionValue,mulAdd_real,ordered_add,normed_real,add_comm]

theorem group_fold_real (plan : Plan N B) (x dy : Nat → Nat → ℝ)
    (stats : Nat → Reuse.Statistics ℝ) (column : Nat) (rows : List Nat) (a : Option (ℝ×ℝ)) :
    optionValue (rows.foldl (fun acc row => groupStep ops plan x dy stats column row acc) a)=
      ((optionValue a).1+(rows.map (fun row => dy row column*((x row column-(stats row).mean)*(stats row).inverse))).sum,
       (optionValue a).2+(rows.map (fun row => dy row column)).sum) := by
  induction rows generalizing a with
  | nil => simp
  | cons row rows ih =>
    simp only [List.foldl_cons,ih,groupStep_real,List.map_cons,List.sum_cons]
    congr 1 <;> ring

theorem group_real (plan : Plan N B) (M G group : Nat) (x dy : Nat → Nat → ℝ)
    (stats : Nat → Reuse.Statistics ℝ) (column : Nat) :
    groupValue ops plan M G group x dy stats column=
      (((groupRows M G group).map (fun row => dy row column*((x row column-(stats row).mean)*(stats row).inverse))).sum,
       ((groupRows M G group).map (fun row => dy row column)).sum) := by
  change optionValue (_)=_
  rw [group_fold_real]
  simp [optionValue]

#axiomsClean forward_real
#axiomsClean backward_real
#axiomsClean group_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.RealProof
