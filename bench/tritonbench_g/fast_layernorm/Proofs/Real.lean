import bench.tritonbench_g.fast_layernorm.Proofs.Model

/-! Independent logical LayerNorm and its first-gradient formulas. Real
sqrt/inverse/division are total. No positive-epsilon, finite-only, nonempty-row
or floating-codec restriction is imposed on this mathematical interpretation. -/
namespace VeriTile.Bench.Repairs.FastLayernorm.RealProof
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.LayerNormFwd.Whole
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 2000000

def ops : Ops ℝ := ⟨0,fun n => n,(·+·),(·-·),(·*·),(·/·),(fun x => -x),fun a b c => a*b+c,fun x => 1/Real.sqrt x⟩

def meanSpec (N : Nat) (x : Nat → ℝ) : ℝ := (∑ j ∈ Finset.range N,x j)/(N : ℝ)
def varianceSpec (N : Nat) (x : Nat → ℝ) : ℝ := (∑ j ∈ Finset.range N,(x j-meanSpec N x)^2)/(N : ℝ)
def invSpec (N : Nat) (x : Nat → ℝ) (eps : ℝ) : ℝ := 1/Real.sqrt (varianceSpec N x+eps)
def outputSpec (N : Nat) (x w bias : Nat → ℝ) (eps : ℝ) (col : Nat) : ℝ :=
  ((x col-meanSpec N x)*invSpec N x eps)*w col+bias col

def gradientSpec (N : Nat) (x dy w : Nat → ℝ) (avg inv : ℝ) (col : Nat) : ℝ :=
  let q := fun j => dy j*w j
  let z := fun j => (x j-avg)*inv
  (q col-(∑ j ∈ Finset.range N,q j)/(N : ℝ)-
    z col*(∑ j ∈ Finset.range N,q j*z j)/(N : ℝ))*inv

def weightGradientSpec (N rows : Nat) (x dy : Nat → Nat → ℝ)
    (stats : Nat → Statistics ℝ) (col : Nat) : ℝ :=
  if col<N then ∑ row ∈ Finset.range rows,dy row col*((x row col-(stats row).mean)*(stats row).inverse) else 0

def biasGradientSpec (N rows : Nat) (dy : Nat → Nat → ℝ) (col : Nat) : ℝ :=
  if col<N then ∑ row ∈ Finset.range rows,dy row col else 0

theorem ordered_add (swap : Bool) (a b : ℝ) : ordered ops.add swap a b=a+b := by
  cases swap <;> simp [ordered,ops,add_comm]
theorem ordered_mul (swap : Bool) (a b : ℝ) : ordered ops.mul swap a b=a*b := by
  cases swap <;> simp [ordered,ops,mul_comm]
theorem mulAdd_real (kind : MulAdd) (a b c : ℝ) : mulAdd ops kind a b c=a*b+c := by
  cases kind with
  | separate sm sa => rw [mulAdd,ordered_add,ordered_mul]
  | fused sm => cases sm <;> simp [mulAdd,ops,mul_comm]
theorem mulSub_real (kind : MulSub) (a b c : ℝ) : mulSub ops kind a b c=a*b-c := by
  cases kind with
  | separate sm => simp only [mulSub,ordered_mul]; rfl
  | fused sm => cases sm <;> simp [mulSub,ops,sub_eq_add_neg,mul_comm]

theorem divide_real (plan : Plan N B) (x : ℝ) : divideCount ops plan x=x/(N : ℝ) := by
  unfold divideCount
  split
  · rename_i h
    rw [plan.eraseUnitValid h]
    simp
  · rfl

theorem sum_padded (tree : SumTree B) (complete : tree.leaves.Perm (List.finRange B))
    (covers : N≤B) (x : Nat → ℝ) :
    tree.eval (·+·) (fun j => if j.val<N then x j.val else 0)=∑ j ∈ Finset.range N,x j := by
  rw [sumTree_real,(complete.map _).sum_eq,←List.ofFn_eq_map,List.sum_ofFn]
  rw [Fin.sum_univ_eq_sum_range (fun j => if j<N then x j else 0)]
  exact VeriTile.Bench.TritonBenchG.LayerNormFwd.Formula.sum_padded x covers

theorem mean_real (plan : Plan N B) (x : Nat → ℝ) (lane : Nat) : mean ops plan x lane=meanSpec N x := by
  unfold mean
  rw [divide_real]
  change (plan.meanTree lane).eval (·+·) (fun j => if j.val<N then x j.val else 0)/(N : ℝ)=_
  rw [sum_padded _ (plan.meanComplete lane) plan.covers]
  rfl

theorem variance_real (plan : Plan N B) (x : Nat → ℝ) (lane : Nat) :
    variance ops plan x lane=varianceSpec N x := by
  have hc : centered ops plan x = fun j : Fin B => if j.val<N then x j.val-meanSpec N x else 0 := by
    funext j
    simp only [centered,mean_real,cast32]
    rfl
  unfold variance
  rw [divide_real,hc]
  let reductions : Reductions B := ⟨plan.meanTree lane,plan.squareTree lane,plan.meanComplete lane,plan.squareComplete lane⟩
  change reductions.variance.eval (·*·) (·+·) (fun a b c => a*b+c) _/(N : ℝ)=_
  rw [variance_padded reductions (fun j => x j-meanSpec N x) plan.covers]
  rfl

theorem statistics_real (plan : Plan N B) (x : Nat → ℝ) (eps : ℝ) (lane : Nat) :
    statistics ops plan x eps lane=⟨meanSpec N x,invSpec N x eps⟩ := by
  simp only [statistics,mean_real,cast32]
  split
  · rename_i h
    have hn := plan.unitVarianceValid h
    subst N
    simp [meanSpec,invSpec,varianceSpec,ops]
  · rw [ordered_add,variance_real]
    rfl

theorem normed_real (plan : Plan N B) (stats : Statistics ℝ) (x : ℝ) :
    normed ops plan stats x=(x-stats.mean)*stats.inverse := by
  unfold normed
  rw [ordered_mul]
  rfl

theorem forward_real (plan : Plan N B) (x w bias : Nat → ℝ) (eps : ℝ) (col : Nat) :
    forwardReference ops plan x w bias eps col=outputSpec N x w bias eps col := by
  unfold forwardReference forwardValue
  rw [statistics_real]
  simp only [cast32,mulAdd_real,normed_real]
  rfl

theorem productTree_real (tree : ProductTree B) (x y : Fin B → ℝ) :
    tree.eval ops x y=(tree.leaves.map (fun j => x j*y j)).sum := by
  induction tree with
  | product j swap => simp [ProductTree.eval,ProductTree.leaves,ordered_mul]
  | add a b ia ib =>
    simp only [ProductTree.eval,ProductTree.leaves,ia,ib,List.map_append,List.sum_append]
    rfl
  | fused j swap rest ih =>
    cases swap <;> simp only [ProductTree.eval,Bool.false_eq_true,if_false,if_true,ih,ProductTree.leaves,List.map_cons,List.sum_cons]
    · rfl
    · simp [ops,mul_comm]

theorem product_padded (tree : ProductTree B) (complete : tree.leaves.Perm (List.finRange B))
    (covers : N≤B) (x y : Nat → ℝ) :
    tree.eval ops (masked ops N x) (masked ops N y)=∑ j ∈ Finset.range N,x j*y j := by
  rw [productTree_real,(complete.map _).sum_eq,←List.ofFn_eq_map,List.sum_ofFn]
  have hp : (fun j : Fin B => masked ops N x j*masked ops N y j)=
      fun j : Fin B => if j.val<N then x j.val*y j.val else 0 := by
    funext j
    by_cases h : j.val<N <;> simp [masked,h,cast32,ops]
  rw [hp,Fin.sum_univ_eq_sum_range (fun j => if j<N then x j*y j else 0)]
  exact VeriTile.Bench.TritonBenchG.LayerNormFwd.Formula.sum_padded (fun j => x j*y j) covers

theorem correction_real (plan : Plan N B) (first z second : ℝ) :
    correctionSubtract ops plan first z second=first-z*second/(N : ℝ) := by
  unfold correctionSubtract
  split
  · rename_i h
    have hn := plan.unitCorrectionValid h
    subst N
    cases plan.correctionSwap <;> simp [ops] <;> ring
  · rw [divide_real,ordered_mul]
    rfl

theorem backward_real (plan : Plan N B) (stats : Statistics ℝ) (x dy w : Nat → ℝ) (col : Nat) :
    backwardReference ops plan stats x dy w col=gradientSpec N x dy w stats.mean stats.inverse col := by
  have hp : backwardProduct ops plan stats x dy w = fun j : Fin B =>
      if j.val<N then dy j.val*w j.val*((x j.val-stats.mean)*stats.inverse) else 0 := by
    funext j
    simp only [backwardProduct,ordered_mul,normed_real]
    by_cases h : j.val<N <;> simp [dyw,masked,h,ops,cast32]
  simp only [backwardReference,cast32,ordered_mul,correction_real,mulSub_real,normed_real,divide_real,
    product_padded _ (plan.firstComplete col) plan.covers,hp]
  have hs := sum_padded (plan.backwardSecond col) (plan.secondComplete col) plan.covers
    (fun j => dy j*w j*((x j-stats.mean)*stats.inverse))
  change (plan.backwardSecond col).eval ops.add _=_ at hs
  rw [hs]
  rfl

theorem affine_real (plan : Plan N B) (rows : Nat) (x dy : Nat → Nat → ℝ)
    (stats : Nat → Statistics ℝ) (col : Nat) :
    affineReference ops plan rows x dy stats col=
      (weightGradientSpec N rows x dy stats col,biasGradientSpec N rows dy col) := by
  induction rows with
  | zero => simp [affineReference,weightGradientSpec,biasGradientSpec,ops]
  | succ rows ih =>
    unfold affineReference at *
    rw [List.range_succ,List.foldl_append]
    change affineStep ops plan x dy stats col rows _=_
    rw [ih]
    by_cases active : col<N
    · simp only [affineStep,active,if_true,cast32,mulAdd_real,normed_real]
      simp only [weightGradientSpec,biasGradientSpec,active,if_true,Finset.sum_range_succ,ops]
      congr 1 <;> ring
    · simp [affineStep,active,ops,weightGradientSpec,biasGradientSpec]

#axiomsClean mean_real
#axiomsClean variance_real
#axiomsClean statistics_real
#axiomsClean forward_real
#axiomsClean backward_real
#axiomsClean affine_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.RealProof
