import bench.repairs.tritonbench_g.log_softmax.Proofs.Graph
import bench.tritonbench_g.log_softmax.Proofs.Math
import bench.tritonbench_g.layer_norm_fwd.Proofs.WholeReduction
import VeriTile.Triton.Semantics.TileOps

/-! Independent arbitrary-Real mathematics. Negative infinity below is only
the source's masked-load sentinel. This is not a decoder of FP32 NaN/Inf words.
The word execution theorem retains all payloads; exact numerical lifting and
exceptional-value policy are stated separately. -/
namespace VeriTile.Bench.Repairs.LogSoftmax.RealProof
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.TritonBenchG.LayerNormFwd.Whole
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 3000000

def targetForward {N : Nat} (x : Fin N → ℝ) (i : Fin N) : ℝ :=
  x i - Real.log (∑ j, Real.exp (x j))

def targetBackward (N : Nat) (y g : Nat → ℝ) (i : Nat) : ℝ :=
  g i - Real.exp (y i) * ∑ j ∈ Finset.range N, g j

def ops : Ops (WithBot ℝ) where
  zero := some 0
  negativeInfinity := ⊥
  add := WithBot.realAdd
  subtract := WithBot.realSub
  multiply := WithBot.realMul
  maximum := max
  exp := WithBot.realExp
  log := WithBot.realLog
  negate := fun x => WithBot.realSub (some 0) x
  fma := fun a b c => WithBot.realAdd (WithBot.realMul a b) c

theorem tree_leaf_le (tree : SumTree C) (x : Fin C → WithBot ℝ) (j : Fin C)
    (member : j∈tree.leaves) : x j≤tree.eval max x := by
  induction tree with
  | leaf k =>
    have same : j=k := by simpa [SumTree.leaves] using member
    subst j
    exact le_rfl
  | add a b ha hb =>
    rcases List.mem_append.mp member with h|h
    · exact le_trans (ha h) (le_max_left ..)
    · exact le_trans (hb h) (le_max_right ..)

theorem tree_le (tree : SumTree C) (x : Fin C → WithBot ℝ) (v : WithBot ℝ)
    (bound : ∀ j∈tree.leaves,x j≤v) : tree.eval max x≤v := by
  induction tree with
  | leaf j => exact bound j (by simp [SumTree.leaves])
  | add a b ha hb =>
    exact max_le (ha (fun j h=>bound j (List.mem_append_left _ h)))
      (hb (fun j h=>bound j (List.mem_append_right _ h)))

theorem max_tree_same (a b : SumTree C) (x : Fin C → WithBot ℝ)
    (ha : a.leaves.Perm (List.finRange C)) (hb : b.leaves.Perm (List.finRange C)) :
    a.eval max x=b.eval max x := by
  apply le_antisymm
  · apply tree_le
    intro j _
    exact tree_leaf_le b x j ((hb.mem_iff).mpr (List.mem_finRange j))
  · apply tree_le
    intro j _
    exact tree_leaf_le a x j ((ha.mem_iff).mpr (List.mem_finRange j))

theorem maximum_finite (plan : Plan N C) (positive : 0<N) (x : Nat → ℝ) :
    ∃ m : ℝ, ∀ lane,rowMax ops plan (fun j=>some (x j)) lane=some m := by
  have hc : 0<C := lt_of_lt_of_le positive plan.covers
  let j : Fin C := ⟨0,hc⟩
  have h := tree_leaf_le (plan.maxTree 0) (forwardLoad ops N (fun j=>some (x j))) j
    ((plan.maxComplete 0).mem_iff.mpr (List.mem_finRange j))
  have hne : rowMax ops plan (fun j=>some (x j)) 0 ≠ ⊥ := by
    intro eq
    change _≤rowMax ops plan (fun j=>some (x j)) 0 at h
    rw [eq] at h
    simp [forwardLoad,ops,cast32,j,positive] at h
  obtain ⟨m,hm⟩ := WithBot.ne_bot_iff_exists.mp hne
  refine ⟨m,?_⟩
  intro lane
  exact (max_tree_same (plan.maxTree lane) (plan.maxTree 0) _
    (plan.maxComplete lane) (plan.maxComplete 0)).trans hm.symm

theorem sum_some (tree : SumTree C) (x : Fin C → ℝ) :
    tree.eval WithBot.realAdd (fun j=>some (x j))=some (tree.eval (·+·) x) := by
  induction tree with
  | leaf j => rfl
  | add a b ha hb => simp only [SumTree.eval,ha,hb];rfl

theorem sum_complete (tree : SumTree C) (complete : tree.leaves.Perm (List.finRange C))
    (x : Fin C → ℝ) : tree.eval (·+·) x=∑ j,x j := by
  rw [Whole.sumTree_real,(complete.map x).sum_eq,←List.ofFn_eq_map,List.sum_ofFn]

theorem sum_prefix (N C : Nat) (covers : N≤C) (x : Nat → ℝ) :
    (∑ j : Fin C,if j.val<N then x j.val else 0)=∑ j : Fin N,x j.val := by
  rw [Fin.sum_univ_eq_sum_range (fun j=>if j<N then x j else 0) C,
    Fin.sum_univ_eq_sum_range x N]
  exact VeriTile.Bench.TritonBenchG.LayerNormFwd.Formula.sum_padded x covers

theorem exponential_values (plan : Plan N C) (x : Nat → ℝ) (m : ℝ)
    (hm : ∀ lane,rowMax ops plan (fun j=>some (x j)) lane=some m) (j : Fin C) :
    exponentials ops plan (fun k=>some (x k)) j=
      some (if j.val<N then Real.exp (x j.val-m) else 0) := by
  simp only [exponentials,shifted,hm,forwardLoad,cast32]
  split_ifs <;> rfl

theorem denominator_value (plan : Plan N C) (x : Nat → ℝ) (m : ℝ)
    (hm : ∀ lane,rowMax ops plan (fun j=>some (x j)) lane=some m) (lane : Nat) :
    denominator ops plan (fun j=>some (x j)) lane=some (∑ j : Fin N,Real.exp (x j.val-m)) := by
  unfold denominator
  rw [show exponentials ops plan (fun j=>some (x j))=
    (fun j : Fin C=>some (if j.val<N then Real.exp (x j.val-m) else 0)) from funext (exponential_values plan x m hm)]
  change (plan.forwardSum lane).eval WithBot.realAdd _=_
  rw [sum_some,sum_complete _ (plan.forwardComplete lane),
    sum_prefix N C plan.covers (fun j=>Real.exp (x j-m))]

theorem stable_target (x : Fin N → ℝ) (m : ℝ) (i : Fin N) :
    (x i-m)-Real.log (∑ j,Real.exp (x j-m))=targetForward x i := by
  unfold targetForward
  rw [←VeriTile.Bench.TritonBenchG.LogSoftmax.Math.forward_logsumexp]
  rw [←VeriTile.Bench.TritonBenchG.LogSoftmax.Math.forward_shift x m i]
  rw [Real.log_div (Real.exp_ne_zero _)
    (ne_of_gt (VeriTile.Bench.TritonBenchG.LogSoftmax.Math.denominator_positive (fun j=>x j-m) i)),Real.log_exp]

theorem forward_target (plan : Plan N C) (x : Nat → ℝ) (i : Fin N) :
    forward ops plan (fun j=>some (x j)) i.val=
      some (targetForward (fun j : Fin N=>x j.val) i) := by
  obtain ⟨m,hm⟩ := maximum_finite plan (Nat.zero_lt_of_lt i.isLt) x
  unfold forward
  rw [hm,denominator_value plan x m hm]
  change some ((x i.val-m)-Real.log (∑ j : Fin N,Real.exp (x j.val-m)))=_
  rw [stable_target (fun j : Fin N=>x j.val) m i]

theorem scale_value (plan : Plan N C) (g : Nat → ℝ) (lane : Nat) :
    scale ops plan (fun j=>some (g j)) lane=some (∑ j ∈ Finset.range N,g j) := by
  unfold scale
  have pointwise : gradientLoad ops N (fun j=>some (g j))=
      (fun j : Fin C=>some (if j.val<N then g j.val else 0)) := by
    funext j
    simp only [gradientLoad,cast32]
    split_ifs <;> rfl
  rw [pointwise]
  change (plan.backwardSum lane).eval WithBot.realAdd _=_
  rw [sum_some,sum_complete _ (plan.backwardComplete lane),sum_prefix N C plan.covers,
    Fin.sum_univ_eq_sum_range]

theorem backward_target (plan : Plan N C) (y g : Nat → ℝ) (i : Nat) :
    backward ops plan (fun j=>some (y j)) (fun j=>some (g j)) i=some (targetBackward N y g i) := by
  unfold backward
  rw [scale_value]
  cases h : plan.backwardSchedule with
  | separate => rfl
  | fused =>
    change some ((0-Real.exp (y i))*(∑ j ∈ Finset.range N,g j)+g i)=_
    congr 1
    unfold targetBackward
    ring
  | fusedScale =>
    change some ((0-(∑ j ∈ Finset.range N,g j))*Real.exp (y i)+g i)=_
    congr 1
    unfold targetBackward
    ring

#axiomsClean maximum_finite
#axiomsClean forward_target
#axiomsClean backward_target
end
end VeriTile.Bench.Repairs.LogSoftmax.RealProof
