import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.RawObservedIO

namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RawObserved
open VeriTile.Triton FP.Structural Math Program Backend
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 3000000

def realModel (revBias : Bool) : NativeModel ℝ where
  biasReverse := revBias
  literal := id
  cast := fun _ _ => id
  add := (· + ·)
  mul := (· * ·)
  maximum := max
  exp := Real.exp
  log := Real.log
  minimum := min
  lessEqual := fun x y => decide (x ≤ y)
  fma := fun x y z => x*y+z
  unused := 0

theorem real_min (x y : ℝ) : (if x < y then x else y) = min x y := by
  split_ifs with h
  · exact (min_eq_left (le_of_lt h)).symm
  · exact (min_eq_right (le_of_not_gt h)).symm

theorem real_min_max (x lo hi : ℝ) :
    (if x < hi ∧ lo < hi then max x lo else hi) = min (max x lo) hi := by
  simpa only [max_lt_iff] using real_min (max x lo) hi

theorem preparation_real (revBias hasBias useSoftplus active : Bool)
    (lower upper : ℝ) (x bias : ℝ) :
    referencePrepared (realModel revBias) hasBias useSoftplus lower upper active x bias =
      if active then min (max
        (if useSoftplus then Math.softplus (x+if hasBias then bias else 0)
         else x+if hasBias then bias else 0) lower) upper else 0 := by
  cases revBias <;> cases hasBias <;> cases useSoftplus <;> cases active
  all_goals simp [referencePrepared,realModel,Math.softplus,UpperBound.cap,add_comm]
  all_goals norm_num


theorem logical_preparation_real (revBias : Bool) (c : Whole.Host) (a : InputWords ℝ)
    (hasBias useSoftplus : Bool) (lower upper : ℝ) (b h k j : Nat) :
    referenceLogicalPrepared (realModel revBias) c a hasBias useSoftplus lower upper b h k j =
      Whole.logicalPrepared c ⟨a.dt,a.rate,a.bias⟩ hasBias useSoftplus lower (.finite upper) b h k j := by
  by_cases ha : h < c.heads ∧ j < c.chunk ∧ k*c.chunk+j < c.sequence
  · simp only [referenceLogicalPrepared,Whole.logicalPrepared,ha,if_true,preparation_real]
    rfl
  · norm_num [referenceLogicalPrepared,Whole.logicalPrepared,ha,realModel]

theorem tree_real (revBias : Bool) (f : Nat → ℝ) (rate : ℝ) (tree : SumTree n) :
    referenceTree (realModel revBias) (fun i => f i.val) rate tree =
      (tree.leaves.map (fun j => f j*rate)).sum := by
  induction tree with
  | leaf i => simp [referenceTree,SumTree.leaves,realModel]
  | add l r il ir =>
    change referenceTree (realModel revBias) (fun i => f i.val) rate l +
      referenceTree (realModel revBias) (fun i => f i.val) rate r = _
    rw [il,ir]
    simp [SumTree.leaves]
  | fused i r ih =>
    change f i.val*rate + referenceTree (realModel revBias) (fun i => f i.val) rate r = _
    rw [ih]
    simp [SumTree.leaves]

theorem list_range_sum (f : Nat → ℝ) (n : Nat) :
    ((List.range n).map f).sum = ∑ j ∈ Finset.range n, f j := by
  induction n with
  | zero => simp
  | succ n ih => simp [List.range_succ,Finset.sum_range_succ,ih]

/-- The exact-real projection of the observed ordered mixed graph is the
independent inclusive mathematical prefix. This theorem does not identify
FP32 evaluation with Real or use an FP reassociation/commutation premise. -/
theorem cumulative_real (revBias : Bool) (c : Whole.Host) (plan : PrefixPlan c.BH c.BC)
    (a : InputWords ℝ) (hasBias useSoftplus : Bool) (lower upper : ℝ)
    (b h k : Nat) (j : Fin c.BC) :
    referenceLogicalCumulative (realModel revBias) c plan a hasBias useSoftplus lower upper b h k j =
      Whole.logicalCumulative c ⟨a.dt,a.rate,a.bias⟩ hasBias useSoftplus lower (.finite upper) b h k j.val := by
  change referenceTree (realModel revBias)
    (fun t => referenceLogicalPrepared (realModel revBias) c a hasBias useSoftplus lower upper b h k t.val)
    (a.rate (h*c.rateHead)) (plan.tree ⟨h%c.BH,Nat.mod_lt h c.headPositive⟩ j) = _
  simp only [logical_preparation_real]
  rw [tree_real]
  have hp := (plan.leaves ⟨h%c.BH,Nat.mod_lt h c.headPositive⟩ j).map
    (fun t => Whole.logicalPrepared c ⟨a.dt,a.rate,a.bias⟩ hasBias useSoftplus lower (.finite upper) b h k t * a.rate (h*c.rateHead))
  rw [hp.sum_eq,list_range_sum]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RawObserved
