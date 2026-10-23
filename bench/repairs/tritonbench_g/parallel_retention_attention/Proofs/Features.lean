import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.QueryReal

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention
noncomputable section
open scoped BigOperators

theorem feature_dot (n b : Nat) (hb : 0<b) (q k : Nat → ℝ) (scale : ℝ) :
    (∑ shard∈Finset.range (blocks n b),∑ e∈Finset.range b,
      ((if shard*b+e<n then q (shard*b+e) else 0)*scale)*
        (if shard*b+e<n then k (shard*b+e) else 0))=
      scale*∑ e∈Finset.range n,q e*k e := by
  calc
    _ = ∑ shard∈Finset.range (blocks n b),∑ e∈Finset.range b,
        if shard*b+e<n then scale*(q (shard*b+e)*k (shard*b+e)) else 0 := by
      apply Finset.sum_congr rfl
      intro shard hshard
      apply Finset.sum_congr rfl
      intro e he
      split_ifs <;> ring
    _ = ∑ e∈Finset.range n,scale*(q e*k e) := tiled_partition n b hb (fun e => scale*(q e*k e))
    _ = _ := (Finset.mul_sum _ _ _).symm

def forwardShards (p : Plans) (s : Shape) (x : Inputs ℝ) (bh t v : Nat) : ℝ :=
  ∑ shard∈Finset.range (blocks s.DK s.BK),
    forwardLane realOps p s x ⟨bh,t/128,shard,v/s.BV⟩ (t%128) (v%s.BV)

def queryShards (p : Plans) (s : Shape) (x : Inputs ℝ) (bh t e : Nat) : ℝ :=
  ∑ shard∈Finset.range (blocks s.DV s.BV),
    queryLane realOps p s x ⟨bh,t/64,e/s.BK,shard⟩ (t%64) (e%s.BK)

theorem forward_full_features (p : Plans) (s : Shape) (x : Inputs ℝ) (bh t v : Nat)
    (hbk : 0<s.BK) (hbv : 0<s.BV) (ht : t<s.T) (hv : v<s.DV) :
    forwardShards p s x bh t v=(Real.sqrt (s.DK:ℝ))⁻¹*
      ∑ n∈Finset.range (t+1),gamma (bh%s.H)^(t-n)*
        (∑ e∈Finset.range s.DK,x.q bh t e*x.k bh n e)*x.v bh n v := by
  have time : t/128*128+t%128=t := by omega
  have col : v/s.BV*s.BV+v%s.BV=v := by simpa [Nat.mul_comm] using Nat.div_add_mod v s.BV
  unfold forwardShards
  simp_rw [forward_causal _ _ _ _ _ _ (Nat.mod_lt t (by norm_num : 0<128))]
  simp only [time]
  rw [Finset.sum_comm,Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro n hn
  have nt : n<s.T := by have := Finset.mem_range.mp hn;omega
  unfold forwardProduct
  simp only [time,col,load,ht,nt,hv,true_and,and_self,if_true]
  calc
    _ = gamma (bh%s.H)^(t-n)*
        (∑ shard∈Finset.range (blocks s.DK s.BK),∑ e∈Finset.range s.BK,
          ((if shard*s.BK+e<s.DK then x.q bh t (shard*s.BK+e) else 0)*(Real.sqrt (s.DK:ℝ))⁻¹)*
          (if shard*s.BK+e<s.DK then x.k bh n (shard*s.BK+e) else 0))*x.v bh n v := by
      simp only [Finset.mul_sum,Finset.sum_mul]
      apply Finset.sum_congr rfl
      intro shard hs
      apply Finset.sum_congr rfl
      intro e he
      simp only [real_zero,mul_assoc]
    _ = _ := by rw [feature_dot _ _ hbk];ring

theorem query_full_features (p : Plans) (s : Shape) (x : Inputs ℝ) (bh t e : Nat)
    (hbk : 0<s.BK) (hbv : 0<s.BV) (ht : t<s.T) (he : e<s.DK) :
    queryShards p s x bh t e=(Real.sqrt (s.DK:ℝ))⁻¹*
      ∑ n∈Finset.range (t+1),gamma (bh%s.H)^(t-n)*x.k bh n e*
        (∑ v∈Finset.range s.DV,x.g bh t v*x.v bh n v) := by
  have time : t/64*64+t%64=t := by omega
  have col : e/s.BK*s.BK+e%s.BK=e := by simpa [Nat.mul_comm] using Nat.div_add_mod e s.BK
  unfold queryShards
  simp_rw [query_causal _ _ _ _ _ _ (Nat.mod_lt t (by norm_num : 0<64))]
  simp only [time]
  rw [← Finset.mul_sum,Finset.sum_comm]
  congr 1
  apply Finset.sum_congr rfl
  intro n hn
  have nt : n<s.T := by have := Finset.mem_range.mp hn;omega
  unfold queryProduct
  simp only [time,col,load,ht,nt,he,true_and,and_self,if_true]
  have hsum := feature_dot s.DV s.BV hbv (x.g bh t) (x.v bh n) 1
  simp only [mul_one,one_mul] at hsum
  calc
    _ = gamma (bh%s.H)^(t-n)*
        (∑ shard∈Finset.range (blocks s.DV s.BV),∑ v∈Finset.range s.BV,
          (if shard*s.BV+v<s.DV then x.g bh t (shard*s.BV+v) else 0)*
          (if shard*s.BV+v<s.DV then x.v bh n (shard*s.BV+v) else 0))*x.k bh n e := by
      simp only [Finset.mul_sum,Finset.sum_mul]
      apply Finset.sum_congr rfl
      intro shard hs
      apply Finset.sum_congr rfl
      intro v hv
      simp only [real_zero,mul_assoc]
    _ = _ := by rw [hsum];ring

#axiomsClean feature_dot
#axiomsClean forward_full_features
#axiomsClean query_full_features
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention
