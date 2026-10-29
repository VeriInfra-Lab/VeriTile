import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.Features
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.ReverseCausal

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention
noncomputable section
open scoped BigOperators

def keyShards (p : Plans) (s : Shape) (x : Inputs ℝ) (bh n e : Nat) : ℝ :=
  ∑ shard∈Finset.range (blocks s.DV s.BV),
    (keyValueLane realOps p s x ⟨bh,n/64,e/s.BK,shard⟩ (n%64) (e%s.BK) 0).1

def valueShards (p : Plans) (s : Shape) (x : Inputs ℝ) (bh n v : Nat) : ℝ :=
  ∑ shard∈Finset.range (blocks s.DK s.BK),
    (keyValueLane realOps p s x ⟨bh,n/64,shard,v/s.BV⟩ (n%64) 0 (v%s.BV)).2

theorem key_full_features (p : Plans) (s : Shape) (x : Inputs ℝ) (bh n e : Nat)
    (hbk : 0<s.BK) (hbv : 0<s.BV) (hn : n<s.T) (he : e<s.DK) :
    keyShards p s x bh n e=(Real.sqrt (s.DK:ℝ))⁻¹*
      ∑ t∈Finset.range s.T,if n≤t then gamma (bh%s.H)^(t-n)*x.q bh t e*
        (∑ v∈Finset.range s.DV,x.g bh t v*x.v bh n v) else 0 := by
  have time : n/64*64+n%64=n := by omega
  have col : e/s.BK*s.BK+e%s.BK=e := by simpa [Nat.mul_comm] using Nat.div_add_mod e s.BK
  unfold keyShards
  simp_rw [keyValue_causal _ _ _ _ _ _ _ (Nat.mod_lt n (by norm_num : 0<64))]
  simp only [Prod.fst,time]
  rw [← Finset.mul_sum,Finset.sum_comm]
  congr 1
  apply Finset.sum_congr rfl
  intro t ht
  have htime : t<s.T := Finset.mem_range.mp ht
  by_cases hc : n≤t
  · simp only [if_pos hc]
    unfold keyProduct
    simp only [time,col,load,hn,htime,he,true_and,and_self,if_true]
    have hsum := feature_dot s.DV s.BV hbv (x.v bh n) (x.g bh t) 1
    simp only [mul_one,one_mul] at hsum
    calc
      _ = gamma (bh%s.H)^(t-n)*
          (∑ shard∈Finset.range (blocks s.DV s.BV),∑ v∈Finset.range s.BV,
            (if shard*s.BV+v<s.DV then x.v bh n (shard*s.BV+v) else 0)*
            (if shard*s.BV+v<s.DV then x.g bh t (shard*s.BV+v) else 0))*x.q bh t e := by
        simp only [Finset.mul_sum,Finset.sum_mul]
        apply Finset.sum_congr rfl
        intro shard hs
        apply Finset.sum_congr rfl
        intro v hv
        simp only [real_zero,mul_assoc]
      _ = _ := by
        rw [hsum,Finset.sum_congr rfl (fun v _ => mul_comm (x.v bh n v) (x.g bh t v))]
        ring
  · simp [hc]

theorem value_full_features (p : Plans) (s : Shape) (x : Inputs ℝ) (bh n v : Nat)
    (hbk : 0<s.BK) (hbv : 0<s.BV) (hn : n<s.T) (hv : v<s.DV) :
    valueShards p s x bh n v=(Real.sqrt (s.DK:ℝ))⁻¹*
      ∑ t∈Finset.range s.T,if n≤t then gamma (bh%s.H)^(t-n)*x.g bh t v*
        (∑ e∈Finset.range s.DK,x.q bh t e*x.k bh n e) else 0 := by
  have time : n/64*64+n%64=n := by omega
  have col : v/s.BV*s.BV+v%s.BV=v := by simpa [Nat.mul_comm] using Nat.div_add_mod v s.BV
  unfold valueShards
  simp_rw [keyValue_causal _ _ _ _ _ _ _ (Nat.mod_lt n (by norm_num : 0<64))]
  simp only [Prod.snd,time]
  rw [← Finset.mul_sum,Finset.sum_comm]
  congr 1
  apply Finset.sum_congr rfl
  intro t ht
  have htime : t<s.T := Finset.mem_range.mp ht
  by_cases hc : n≤t
  · simp only [if_pos hc]
    unfold valueProduct
    simp only [time,col,load,hn,htime,hv,true_and,and_self,if_true]
    have hsum := feature_dot s.DK s.BK hbk (x.k bh n) (x.q bh t) 1
    simp only [mul_one,one_mul] at hsum
    calc
      _ = gamma (bh%s.H)^(t-n)*
          (∑ shard∈Finset.range (blocks s.DK s.BK),∑ e∈Finset.range s.BK,
            (if shard*s.BK+e<s.DK then x.k bh n (shard*s.BK+e) else 0)*
            (if shard*s.BK+e<s.DK then x.q bh t (shard*s.BK+e) else 0))*x.g bh t v := by
        simp only [Finset.mul_sum,Finset.sum_mul]
        apply Finset.sum_congr rfl
        intro shard hs
        apply Finset.sum_congr rfl
        intro e he
        simp only [real_zero,mul_assoc]
      _ = _ := by
        rw [hsum,Finset.sum_congr rfl (fun e _ => mul_comm (x.k bh n e) (x.q bh t e))]
        ring
  · simp [hc]

#axiomsClean key_full_features
#axiomsClean value_full_features
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention
