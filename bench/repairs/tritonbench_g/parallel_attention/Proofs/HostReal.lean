import bench.repairs.tritonbench_g.parallel_attention.Proofs.ShardReal
import bench.repairs.tritonbench_g.parallel_attention.Proofs.Grid

namespace VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
open scoped BigOperators

def logicalSpec (s : Shape) (kind : OutputKind) (x : TensorInputs ℝ) (bh t e : Nat) : ℝ :=
  let scale := selectedScale s.K s.useScale
  let a := headInputs x bh
  let g := headCotangent x bh
  match kind with
  | .forward => outputSpec s.T s.K scale a t e
  | .normalizer => normalizerSpec s.T s.K scale a t
  | .query => querySpec s.T s.K s.V scale a g t e
  | .key => keySpec s.T s.K s.V scale a g t e
  | .value => valueSpec s.T s.K scale a g t e

def ownedLane (a : Ops α) (p : Plans) (s : Shape) (kind : OutputKind)
    (x : TensorInputs α) (shard bh t e : Nat) : α :=
  match kind with
  | .forward => (forwardLane a p s x ⟨bh,t/128,0,e/s.BV⟩ (t%128) (e%s.BV)).1
  | .normalizer => (forwardLane a p s x ⟨bh,t/128,0,0⟩ (t%128) 0).2
  | .query => queryLane a p s x ⟨bh,t/64,0,shard⟩ (t%64) e
  | .key => (keyValueLane a p s x ⟨bh,t/64,0,shard⟩ (t%64) e 0).1
  | .value => (keyValueLane a p s x ⟨bh,t/64,0,e/s.BV⟩ (t%64) 0 (e%s.BV)).2

theorem scratch_owned (a : Ops α) (p : Plans) (s : Shape) (kind : OutputKind)
    (x : TensorInputs α) (shard bh t e : Nat) (hb : bh < s.B*s.H)
    (ht : t < s.T) (he : e < kind.width s) :
    scratchValue a p s kind x (shard*kind.cells s+dense s.T (kind.width s) bh t e)=
      ownedLane a p s kind x shard bh t e := by
  obtain ⟨h1,h2,h3,h4⟩ := scratch_decode (s.B*s.H) s.T (kind.width s) shard bh t e hb ht he
  unfold scratchValue
  dsimp only
  change (match kind with
    | .forward => _ | .normalizer => _ | .query => _ | .key => _ | .value => _)=_
  simp only [OutputKind.cells,h1,h2,h3,h4]
  rfl

theorem wordReference_spec (p : Plans) (s : Shape) (kind : OutputKind)
    (tree : SumPlan) (complete : tree.Covers (kind.shards s)) (x : TensorInputs ℝ)
    (bh t e : Nat) (hb : bh < s.B*s.H) (ht : t < s.T) (he : e < kind.width s)
    (hK : 0 < s.K) (fits : s.K ≤ s.BK) (hV : 0 < s.V) (hbv : 0 < s.BV) :
    wordReference realOps p s kind tree x (dense s.T (kind.width s) bh t e)=
      logicalSpec s kind x bh t e := by
  have hi : dense s.T (kind.width s) bh t e < kind.cells s :=
    dense_bound _ _ _ _ _ _ hb ht he
  rw [wordReference_real p s kind tree complete x _ hi]
  simp_rw [scratch_owned realOps p s kind x _ bh t e hb ht he]
  have h64 : t/64*64+t%64=t := by omega
  have h128 : t/128*128+t%128=t := by omega
  have hval : e/s.BV*s.BV+e%s.BV=e := by
    simpa [Nat.mul_comm,Nat.add_comm] using Nat.mod_add_div e s.BV
  have hr64 : t%64 < 64 := Nat.mod_lt _ (by norm_num)
  have hr128 : t%128 < 128 := Nat.mod_lt _ (by norm_num)
  cases kind with
  | forward =>
    simp only [OutputKind.shards,Finset.sum_range_one,ownedLane,logicalSpec]
    have h := forward_whole_real p s x ⟨bh,t/128,0,e/s.BV⟩ (t%128) (e%s.BV) rfl fits hr128
      (by simpa only [h128] using ht) (by simpa only [hval] using he)
    simpa only [h128,hval] using congrArg Prod.fst h
  | normalizer =>
    simp only [OutputKind.shards,Finset.sum_range_one,ownedLane,logicalSpec]
    have h := forward_whole_real p s x ⟨bh,t/128,0,0⟩ (t%128) 0 rfl fits hr128
      (by simpa only [h128] using ht) (by simpa using hV)
    simpa only [h128] using congrArg Prod.snd h
  | query =>
    simp only [OutputKind.shards,ownedLane,logicalSpec]
    simpa only [h64] using query_shards_real p s x bh (t/64) (t%64) e fits hr64
      (by simpa only [h64] using ht) he hbv hV
  | key =>
    simp only [OutputKind.shards,ownedLane,logicalSpec]
    simpa only [h64] using key_shards_real p s x bh (t/64) (t%64) e fits hr64
      (by simpa only [h64] using ht) he hbv hV
  | value =>
    simp only [OutputKind.shards,Finset.sum_range_one,ownedLane,logicalSpec]
    simpa only [h64,hval] using value_lane_spec p s x ⟨bh,t/64,0,e/s.BV⟩ (t%64) (e%s.BV) rfl fits
      (by simpa only [h64] using ht) hK (by simpa only [hval] using he) hr64

/-- Current memory, arbitrary complete store serialization, actual host reduction,
and independent mathematical output meet in this endpoint. -/
theorem whole_real_correct (p : Plans) (s : Shape) (b : Bases) (kind : OutputKind)
    (tree : SumPlan) (complete : tree.Covers (kind.shards s))
    (scratch output : Nat) (schedule : List Nat)
    (covers : schedule.Perm (List.range (kind.count s)))
    (scratch_fresh : ∀ i,i < kind.count s → scratch+i∉inputCells s b)
    (output_fresh : ∀ i,i < kind.cells s → output+i∉scratchCells (kind.shards s) scratch (kind.cells s))
    (m : Memory ℝ) (bh t e : Nat) (hb : bh < s.B*s.H) (ht : t < s.T) (he : e < kind.width s)
    (hK : 0 < s.K) (fits : s.K ≤ s.BK) (hV : 0 < s.V) (hbv : 0 < s.BV) :
    wholeRun realOps p s b kind tree scratch output schedule m
        (output+dense s.T (kind.width s) bh t e)=
      logicalSpec s kind (currentInputs realOps s b m) bh t e := by
  rw [whole_output realOps p s b kind tree scratch output schedule covers scratch_fresh output_fresh m _
    (dense_bound _ _ _ _ _ _ hb ht he)]
  exact wordReference_spec p s kind tree complete _ bh t e hb ht he hK fits hV hbv

#axiomsClean scratch_owned
#axiomsClean wordReference_spec
#axiomsClean whole_real_correct
end
end VeriTile.Bench.Repairs.ParallelAttention
