import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.RealPartitions
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.RealTiles
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.Real
open VeriTile Triton FP.Structural
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

theorem past_partition (N BT block time : Nat) (ht : time/BT=block) (f : Nat → ℝ) :
    (∑ u : Fin N, if u.val/BT=block ∧ u.val ≤ time then f u.val else 0)+
    (∑ u : Fin N, if u.val/BT<block then f u.val else 0)=
      ∑ u : Fin N, if u.val ≤ time then f u.val else 0 := by
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro u _
  simpa only [ht,add_comm] using ChunkLinearAttn.Real.causal_partition BT u.val time (f u.val)

theorem future_partition (N BT block time : Nat) (ht : time/BT=block) (f : Nat → ℝ) :
    (∑ t : Fin N, if t.val/BT=block ∧ time ≤ t.val then f t.val else 0)+
    (∑ t : Fin N, if block<t.val/BT then f t.val else 0)=
      ∑ t : Fin N, if time ≤ t.val then f t.val else 0 := by
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro t _
  simpa only [ht,add_comm,eq_comm] using ChunkLinearAttn.Real.causal_partition BT time t.val (f t.val)

theorem forward_inter (p : Parameters) (hp : 0<p.BT) (pid : PID) (m : Memory ℝ)
    (n : Nat) (i : TileIndex [p.BT,p.BM]) (ht : n*p.BT+i.1.val<p.N) :
    dot dotPrimitive (fwdQ algebra p pid m n) (fwdCarry algebra dotPrimitive p pid m n) i=
      ∑ u : Fin p.N, if u.val/p.BT<n then score p pid.head m (n*p.BT+i.1.val) u.val*
        value p pid.head m u.val (pid.channel*p.BM+i.2.1.val) else 0 := by
  rw [dot_eq]
  simp only [fwdQ,read_eq,ht,ite_true]
  simp_rw [forward_carry p hp,Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro u _
  by_cases hu : u.val/p.BT<n
  · simp only [hu,ite_true]
    simp [score,query,Finset.sum_mul,mul_assoc,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  · simp [hu]

theorem query_inter (p : Parameters) (hp : 0<p.BT) (htile : p.NC*p.CB=p.BT)
    (head : Nat) (m : Memory ℝ) (n time : Nat) (a : Fin p.D) :
    (∑ c : Fin p.E, incoming p head m time c.val*queryCarry algebra dotPrimitive p head m n (c,a,()))=
      ∑ u : Fin p.N, if u.val/p.BT<n then scoreGradient p head m time u.val*key p head m u.val a.val else 0 := by
  simp_rw [query_carry p hp htile,Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro u _
  by_cases hu : u.val/p.BT<n <;> simp [hu,scoreGradient,Finset.sum_mul,mul_assoc]

theorem key_inter (p : Parameters) (hp : 0<p.BT) (htile : p.NC*p.CB=p.BT)
    (hcover : p.N ≤ p.NT*p.BT) (head : Nat) (m : Memory ℝ) (n time : Nat)
    (hn : n<p.NT) (a : Fin p.D) :
    (∑ c : Fin p.E, keyValueCarry algebra dotPrimitive p head m (p.NT-1-n) (a,c,())*value p head m time c.val)=
      ∑ t : Fin p.N, if n<t.val/p.BT then query p head m t.val a.val*scoreGradient p head m t.val time else 0 := by
  simp_rw [key_value_carry p hp htile hcover head (p.NT-1-n) (by omega),Finset.sum_mul]
  rw [Finset.sum_comm]
  have he : p.NT-(p.NT-1-n)=n+1 := by omega
  simp only [he,Nat.add_one_le_iff]
  apply Finset.sum_congr rfl
  intro t _
  by_cases ht : n<t.val/p.BT <;> simp [ht,scoreGradient,Finset.mul_sum,mul_assoc]

theorem value_inter (p : Parameters) (hp : 0<p.BT) (htile : p.NC*p.CB=p.BT)
    (hcover : p.N ≤ p.NT*p.BT) (head : Nat) (m : Memory ℝ) (n time : Nat)
    (hn : n<p.NT) (c : Fin p.E) :
    (∑ a : Fin p.D, key p head m time a.val*keyValueCarry algebra dotPrimitive p head m (p.NT-1-n) (a,c,()))=
      ∑ t : Fin p.N, if n<t.val/p.BT then score p head m t.val time*incoming p head m t.val c.val else 0 := by
  simp_rw [key_value_carry p hp htile hcover head (p.NT-1-n) (by omega),Finset.mul_sum]
  rw [Finset.sum_comm]
  have he : p.NT-(p.NT-1-n)=n+1 := by omega
  simp only [he,Nat.add_one_le_iff]
  apply Finset.sum_congr rfl
  intro t _
  by_cases ht : n<t.val/p.BT <;> simp [ht,score,Finset.sum_mul,Finset.mul_sum,mul_assoc,mul_comm,mul_left_comm]

/-- The actual source carry and masked tile contraction produce every public
forward coordinate of the independent causal target. -/
theorem forward_value (p : Parameters) (hp : 0<p.BT) (hcover : p.N ≤ p.NT*p.BT)
    (pid : PID) (m : Memory ℝ) (n : Nat) (hn : n<p.NT)
    (i : TileIndex [p.BT,p.BM]) (ht : n*p.BT+i.1.val<p.N)
    (hc : pid.channel*p.BM+i.2.1.val<p.E) :
    fwdOutput algebra dotPrimitive p pid m n (fwdCarry algebra dotPrimitive p pid m n) i=
      Target.output (logicalQ p pid.head m) (logicalK p pid.head m) (logicalV p pid.head m)
        ⟨n*p.BT+i.1.val,ht⟩ ⟨pid.channel*p.BM+i.2.1.val,hc⟩ := by
  simp only [fwdOutput,add_eq]
  rw [forward_intra p hp hcover pid m n hn i ht,forward_inter p hp pid m n i ht]
  have h := past_partition p.N p.BT n (n*p.BT+i.1.val)
    (ChunkLinearAttn.Real.block_quotient n p.BT i.1.val i.1.isLt)
    (fun u => score p pid.head m (n*p.BT+i.1.val) u*value p pid.head m u (pid.channel*p.BM+i.2.1.val))
  simpa only [Target.output,Target.score,logicalQ,logicalK,logicalV,score,Fin.le_def] using h

theorem query_value (p : Parameters) (hp : 0<p.BT) (htile : p.NC*p.CB=p.BT)
    (hcover : p.N ≤ p.NT*p.BT) (head : Nat) (m : Memory ℝ) (n : Nat) (hn : n<p.NT)
    (i : TileIndex [p.BT,p.D]) (ht : n*p.BT+i.1.val<p.N) :
    intraDQ algebra dotPrimitive p head n m i+
      (∑ c : Fin p.E, incoming p head m (n*p.BT+i.1.val) c.val*
        queryCarry algebra dotPrimitive p head m n (c,i.2.1,()))=
      Target.dq (logicalK p head m) (logicalV p head m) (logicalG p head m) ⟨n*p.BT+i.1.val,ht⟩ i.2.1 := by
  rw [intra_query p hp hcover head m n hn i ht,query_inter p hp htile]
  have h := past_partition p.N p.BT n (n*p.BT+i.1.val)
    (ChunkLinearAttn.Real.block_quotient n p.BT i.1.val i.1.isLt)
    (fun u => scoreGradient p head m (n*p.BT+i.1.val) u*key p head m u i.2.1.val)
  simpa only [Target.dq,logicalK,logicalV,logicalG,scoreGradient,Fin.le_def] using h

theorem key_value (p : Parameters) (hp : 0<p.BT) (htile : p.NC*p.CB=p.BT)
    (hcover : p.N ≤ p.NT*p.BT) (head : Nat) (m : Memory ℝ) (n : Nat) (hn : n<p.NT)
    (i : TileIndex [p.D,p.BT]) (ht : n*p.BT+i.2.1.val<p.N) :
    intraDKT algebra dotPrimitive p head n m i+
      (∑ c : Fin p.E, keyValueCarry algebra dotPrimitive p head m (p.NT-1-n) (i.1,c,())*
        value p head m (n*p.BT+i.2.1.val) c.val)=
      Target.dk (logicalQ p head m) (logicalV p head m) (logicalG p head m) ⟨n*p.BT+i.2.1.val,ht⟩ i.1 := by
  rw [intra_key p hp hcover head m n hn i ht,key_inter p hp htile hcover head m n _ hn]
  have h := future_partition p.N p.BT n (n*p.BT+i.2.1.val)
    (ChunkLinearAttn.Real.block_quotient n p.BT i.2.1.val i.2.1.isLt)
    (fun t => query p head m t i.1.val*scoreGradient p head m t (n*p.BT+i.2.1.val))
  simpa only [Target.dk,logicalQ,logicalV,logicalG,scoreGradient,Fin.le_def] using h

theorem value_value (p : Parameters) (hp : 0<p.BT) (htile : p.NC*p.CB=p.BT)
    (hcover : p.N ≤ p.NT*p.BT) (head : Nat) (m : Memory ℝ) (n : Nat) (hn : n<p.NT)
    (i : TileIndex [p.BT,p.E]) (ht : n*p.BT+i.1.val<p.N) :
    intraDV algebra dotPrimitive p head n m i+
      (∑ a : Fin p.D, key p head m (n*p.BT+i.1.val) a.val*
        keyValueCarry algebra dotPrimitive p head m (p.NT-1-n) (a,i.2.1,()))=
      Target.dv (logicalQ p head m) (logicalK p head m) (logicalG p head m) ⟨n*p.BT+i.1.val,ht⟩ i.2.1 := by
  rw [intra_value p hp hcover head m n hn i ht,value_inter p hp htile hcover head m n _ hn]
  have h := future_partition p.N p.BT n (n*p.BT+i.1.val)
    (ChunkLinearAttn.Real.block_quotient n p.BT i.1.val i.1.isLt)
    (fun t => score p head m t (n*p.BT+i.1.val)*incoming p head m t i.2.1.val)
  simpa only [Target.dv,Target.score,logicalQ,logicalK,logicalG,score,Fin.le_def] using h

/-- Local source dot readback for a query subblock; the caller supplies the
actual proved prefix carry. No memory/output premise is hidden here. -/
theorem query_sub_contribution (p : Parameters) (head row : Nat) (m : Memory ℝ)
    (carry : Matrix ℝ p.E p.D) (i : TileIndex [p.CB,p.D]) (ht : row+i.1.val<p.N) :
    dot dotPrimitive (subG algebra p head row m) carry i=
      ∑ c : Fin p.E, incoming p head m (row+i.1.val) c.val*carry (c,i.2.1,()) := by
  simp [dot_eq,subG,ht,incoming]

theorem key_sub_contribution (p : Parameters) (head row : Nat) (m : Memory ℝ)
    (carry : Matrix ℝ p.D p.E) (i : TileIndex [p.D,p.CB]) (ht : row+i.2.1.val<p.N) :
    dot dotPrimitive carry (subVT algebra p head row m) i=
      ∑ c : Fin p.E, carry (i.1,c,())*value p head m (row+i.2.1.val) c.val := by
  simp [dot_eq,subVT,ht,value]

theorem value_sub_contribution (p : Parameters) (head row : Nat) (m : Memory ℝ)
    (carry : Matrix ℝ p.D p.E) (i : TileIndex [p.CB,p.E]) (ht : row+i.1.val<p.N) :
    dot dotPrimitive (subK algebra p head row m) carry i=
      ∑ a : Fin p.D, key p head m (row+i.1.val) a.val*carry (a,i.2.1,()) := by
  simp [dot_eq,subK,ht,key]

#axiomsClean query_sub_contribution
#axiomsClean key_sub_contribution
#axiomsClean value_sub_contribution
#axiomsClean forward_value
#axiomsClean query_value
#axiomsClean key_value
#axiomsClean value_value
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.Real
