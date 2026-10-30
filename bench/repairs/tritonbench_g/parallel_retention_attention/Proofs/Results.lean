import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.BackwardFeatures
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.Grid
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.Differential

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention
noncomputable section
open scoped BigOperators

def realShards (p : Plans) (s : Shape) (kind : OutputKind) (x : Inputs ℝ) (bh t e : Nat) : ℝ :=
  match kind with
  | .forward => forwardShards p s x bh t e
  | .query => queryShards p s x bh t e
  | .key => keyShards p s x bh t e
  | .value => valueShards p s x bh t e

/-- Independent logical-dimensional formulas; no tile size, recurrence,
launch order, or reduction tree occurs in this target. -/
def logicalSpec (s : Shape) (kind : OutputKind) (x : Inputs ℝ) (bh t e : Nat) : ℝ :=
  (Real.sqrt (s.DK:ℝ))⁻¹ * match kind with
  | .forward => ∑ n∈Finset.range s.T,if n≤t then gamma (bh%s.H)^(t-n)*
      (∑ f∈Finset.range s.DK,x.q bh t f*x.k bh n f)*x.v bh n e else 0
  | .query => ∑ n∈Finset.range s.T,if n≤t then gamma (bh%s.H)^(t-n)*x.k bh n e*
      (∑ f∈Finset.range s.DV,x.g bh t f*x.v bh n f) else 0
  | .key => ∑ n∈Finset.range s.T,if t≤n then gamma (bh%s.H)^(n-t)*x.q bh n e*
      (∑ f∈Finset.range s.DV,x.g bh n f*x.v bh t f) else 0
  | .value => ∑ n∈Finset.range s.T,if t≤n then gamma (bh%s.H)^(n-t)*x.g bh n e*
      (∑ f∈Finset.range s.DK,x.q bh n f*x.k bh t f) else 0

theorem realShards_correct (p : Plans) (s : Shape) (kind : OutputKind) (x : Inputs ℝ)
    (bh t e : Nat) (hbk : 0<s.BK) (hbv : 0<s.BV) (ht : t<s.T) (he : e<kind.width s) :
    realShards p s kind x bh t e=logicalSpec s kind x bh t e := by
  cases kind with
  | forward =>
    rw [realShards,forward_full_features p s x bh t e hbk hbv ht he]
    unfold logicalSpec
    congr 1
    simpa only [mul_assoc] using causal_to_logical (gamma (bh%s.H))
      (fun n => (∑ f∈Finset.range s.DK,x.q bh t f*x.k bh n f)*x.v bh n e) s.T t ht
  | query =>
    rw [realShards,query_full_features p s x bh t e hbk hbv ht he]
    unfold logicalSpec
    congr 1
    simpa only [mul_assoc] using causal_to_logical (gamma (bh%s.H))
      (fun n => x.k bh n e*(∑ f∈Finset.range s.DV,x.g bh t f*x.v bh n f)) s.T t ht
  | key => exact key_full_features p s x bh t e hbk hbv ht he
  | value => exact value_full_features p s x bh t e hbk hbv ht he

theorem wordReference_shards (p : Plans) (s : Shape) (kind : OutputKind) (tree : SumPlan)
    (complete : tree.Covers (kind.shards s)) (x : Inputs ℝ) (bh t e : Nat)
    (hb : bh<s.B*s.H) (ht : t<s.T) (he : e<kind.width s) :
    wordReference realOps p s kind tree x (dense s.T (kind.width s) bh t e)=
      realShards p s kind x bh t e := by
  rw [wordReference_real p s kind tree complete x _ (dense_bound _ _ _ _ _ _ hb ht he)]
  cases kind <;> unfold realShards forwardShards queryShards keyShards valueShards
  all_goals
    apply Finset.sum_congr rfl
    intro shard hshard
    have dec := scratch_decode (s.B*s.H) s.T _ shard bh t e hb ht he
    simp only [OutputKind.width] at dec
    dsimp only [scratchValue,OutputKind.cells,OutputKind.width,OutputKind.shards]
    rw [dec.1,dec.2.1,dec.2.2.1,dec.2.2.2]

theorem whole_real (p : Plans) (s : Shape) (b : Bases) (kind : OutputKind)
    (tree : SumPlan) (complete : tree.Covers (kind.shards s))
    (scratch output : Nat) (schedule : List Nat)
    (covers : schedule.Perm (List.range (kind.count s)))
    (scratch_fresh : ∀ i, i<kind.count s → scratch+i∉inputCells s b)
    (output_fresh : ∀ i, i<kind.cells s → output+i∉scratchCells (kind.shards s) scratch (kind.cells s))
    (m : Memory ℝ) (bh t e : Nat) (hbk : 0<s.BK) (hbv : 0<s.BV)
    (hb : bh<s.B*s.H) (ht : t<s.T) (he : e<kind.width s) :
    wholeRun realOps p s b kind tree scratch output schedule m (output+dense s.T (kind.width s) bh t e)=
      logicalSpec s kind (currentInputs realOps s b m) bh t e := by
  rw [whole_output realOps p s b kind tree scratch output schedule covers scratch_fresh output_fresh m _
    (dense_bound _ _ _ _ _ _ hb ht he),wordReference_shards p s kind tree complete _ bh t e hb ht he]
  exact realShards_correct p s kind _ bh t e hbk hbv ht he

/-- Identification with the independently differentiated finite-dimensional
forward formula. The three gradient formulas use the same finite inputs. -/
theorem logical_forward_spec (s : Shape) (x : Inputs ℝ) (bh : Nat) (t : Fin s.T) (e : Fin s.DV) :
    logicalSpec s .forward x bh t e=outputSpec s.T s.DK s.DV (bh%s.H)
      (fun t e => x.q bh t e) (fun t e => x.k bh t e) (fun t e => x.v bh t e) t e := by
  unfold logicalSpec outputSpec weight
  simp_rw [← Fin.sum_univ_eq_sum_range]
  congr 1
  apply Finset.sum_congr rfl
  intro n hn
  split_ifs <;> ring

theorem logical_query_spec (s : Shape) (x : Inputs ℝ) (bh : Nat) (t : Fin s.T) (e : Fin s.DK) :
    logicalSpec s .query x bh t e=querySpec s.T s.DK s.DV (bh%s.H)
      (fun t e => x.k bh t e) (fun t e => x.v bh t e) (fun t e => x.g bh t e) t e := by
  unfold logicalSpec querySpec weight
  simp_rw [← Fin.sum_univ_eq_sum_range]
  congr 1
  apply Finset.sum_congr rfl
  intro n hn
  split_ifs <;> ring

theorem logical_key_spec (s : Shape) (x : Inputs ℝ) (bh : Nat) (t : Fin s.T) (e : Fin s.DK) :
    logicalSpec s .key x bh t e=keySpec s.T s.DK s.DV (bh%s.H)
      (fun t e => x.q bh t e) (fun t e => x.v bh t e) (fun t e => x.g bh t e) t e := by
  unfold logicalSpec keySpec weight
  simp_rw [← Fin.sum_univ_eq_sum_range]
  congr 1
  apply Finset.sum_congr rfl
  intro n hn
  split_ifs <;> ring

theorem logical_value_spec (s : Shape) (x : Inputs ℝ) (bh : Nat) (t : Fin s.T) (e : Fin s.DV) :
    logicalSpec s .value x bh t e=valueSpec s.T s.DK s.DV (bh%s.H)
      (fun t e => x.q bh t e) (fun t e => x.k bh t e) (fun t e => x.g bh t e) t e := by
  unfold logicalSpec valueSpec weight
  simp_rw [← Fin.sum_univ_eq_sum_range]
  congr 1
  apply Finset.sum_congr rfl
  intro n hn
  split_ifs <;> ring

#axiomsClean realShards_correct
#axiomsClean wordReference_shards
#axiomsClean whole_real
#axiomsClean logical_forward_spec
#axiomsClean logical_query_spec
#axiomsClean logical_key_spec
#axiomsClean logical_value_spec
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention
