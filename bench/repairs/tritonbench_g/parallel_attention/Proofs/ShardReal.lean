import bench.repairs.tritonbench_g.parallel_attention.Proofs.ReverseReal

namespace VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
open scoped BigOperators

/-- Host reduction combines every value shard, and dZ occurs in exactly shard zero. -/
theorem query_shards_real (p : Plans) (s : Shape) (x : TensorInputs ℝ)
    (bh chunk r e : Nat) (fits : s.K ≤ s.BK) (hr : r < 64)
    (ht : chunk*64+r < s.T) (he : e < s.K) (hbv : 0 < s.BV) (hv : 0 < s.V) :
    (∑ shard∈Finset.range (blocks s.V s.BV),queryLane realOps p s x ⟨bh,chunk,0,shard⟩ r e)=
      querySpec s.T s.K s.V (selectedScale s.K s.useScale) (headInputs x bh)
        (headCotangent x bh) (chunk*64+r) e := by
  have lane (shard : Nat) (hs : shard∈Finset.range (blocks s.V s.BV)) :
      queryLane realOps p s x ⟨bh,chunk,0,shard⟩ r e=
        selectedScale s.K s.useScale * ∑ j∈Finset.range s.T,queryProduct s x ⟨bh,chunk,0,shard⟩ r e j :=
    query_whole_real p s x ⟨bh,chunk,0,shard⟩ r e rfl fits hr ht he
  rw [Finset.sum_congr rfl lane]
  rw [← Finset.mul_sum,Finset.sum_comm,Finset.mul_sum]
  unfold querySpec
  apply Finset.sum_congr rfl
  intro j hj
  by_cases causal : j ≤ chunk*64+r
  · simp only [queryProduct,if_pos causal,gradientCoefficient,headInputs,if_pos causal]
    simp only [← Finset.sum_mul,← Finset.mul_sum]
    rw [cotangent_shards s x bh (chunk*64+r) j hbv hv]
    dsimp only [headInputs]
    ring
  · simp [queryProduct,gradientCoefficient,causal]

theorem shard_start_valid (V BV shard : Nat) (hbv : 0 < BV)
    (h : shard < blocks V BV) : shard*BV < V := by
  unfold blocks at h
  have hm : (shard+1)*BV ≤ V+BV-1 := (Nat.le_div_iff_mul_le hbv).mp h
  nlinarith [Nat.sub_le (V+BV) 1,show 1 ≤ V+BV by omega, Nat.sub_add_cancel (show 1 ≤ V+BV by omega)]

theorem key_shards_real (p : Plans) (s : Shape) (x : TensorInputs ℝ)
    (bh chunk r e : Nat) (fits : s.K ≤ s.BK) (hr : r < 64)
    (hn : chunk*64+r < s.T) (he : e < s.K) (hbv : 0 < s.BV) (hv : 0 < s.V) :
    (∑ shard∈Finset.range (blocks s.V s.BV),
      (keyValueLane realOps p s x ⟨bh,chunk,0,shard⟩ r e 0).1)=
      keySpec s.T s.K s.V (selectedScale s.K s.useScale) (headInputs x bh)
        (headCotangent x bh) (chunk*64+r) e := by
  have lane (shard : Nat) (hs : shard∈Finset.range (blocks s.V s.BV)) :
      (keyValueLane realOps p s x ⟨bh,chunk,0,shard⟩ r e 0).1=
        ∑ t∈Finset.range s.T,keyProduct s x ⟨bh,chunk,0,shard⟩ r e t := by
    have h := keyValue_whole_real p s x ⟨bh,chunk,0,shard⟩ r e 0 rfl fits hn he
      (by simpa using shard_start_valid s.V s.BV shard hbv (Finset.mem_range.mp hs)) hr
    exact congrArg Prod.fst h
  rw [Finset.sum_congr rfl lane,Finset.sum_comm]
  unfold keySpec
  apply Finset.sum_congr rfl
  intro t ht
  have hvalid := Finset.mem_range.mp ht
  by_cases causal : chunk*64+r ≤ t
  · simp only [keyProduct,hvalid,causal,and_self,if_true,gradientCoefficient,if_pos causal,headInputs]
    simp only [← Finset.sum_mul,← Finset.mul_sum]
    rw [cotangent_shards s x bh t (chunk*64+r) hbv hv]
    rfl
  · simp [keyProduct,gradientCoefficient,causal]

theorem value_lane_spec (p : Plans) (s : Shape) (x : TensorInputs ℝ)
    (tile : Tile) (r v : Nat) (hk : tile.key=0) (fits : s.K ≤ s.BK)
    (hn : tile.chunk*64+r < s.T) (hK : 0 < s.K)
    (hv : tile.value*s.BV+v < s.V) (hr : r < 64) :
    (keyValueLane realOps p s x tile r 0 v).2=
      valueSpec s.T s.K (selectedScale s.K s.useScale) (headInputs x tile.bh)
        (headCotangent x tile.bh) (tile.chunk*64+r) (tile.value*s.BV+v) := by
  rw [keyValue_whole_real p s x tile r 0 v hk fits hn hK hv hr]
  unfold valueSpec
  dsimp only [Prod.snd]
  apply Finset.sum_congr rfl
  intro t ht
  have hvalid := Finset.mem_range.mp ht
  by_cases causal : tile.chunk*64+r ≤ t <;>
    simp [valueProduct,weight,hvalid,causal,headCotangent]

#axiomsClean query_shards_real
#axiomsClean shard_start_valid
#axiomsClean key_shards_real
#axiomsClean value_lane_spec
end
end VeriTile.Bench.Repairs.ParallelAttention
