import bench.repairs.tritonbench_g.lightning_attention.Proofs.RealInputs
namespace VeriTile.Bench.Repairs.LightningAttention.Real
open VeriTile Triton FP.Structural
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

def score (p : Parameters) (head : Nat) (m : Memory ℝ) (t u : Nat) : ℝ :=
  ∑ a : Fin p.D, query p head m t a.val*key p head m u a.val
def scoreGradient (p : Parameters) (head : Nat) (m : Memory ℝ) (t u : Nat) : ℝ :=
  ∑ c : Fin p.E, incoming p head m t c.val*value p head m u c.val

theorem forward_intra (p : Parameters) (hp : 0<p.BT) (hcover : p.N ≤ p.NT*p.BT)
    (pid : PID) (m : Memory ℝ) (n : Nat) (hn : n<p.NT)
    (i : TileIndex [p.BT,p.BM]) (ht : n*p.BT+i.1.val<p.N) :
    dot dotPrimitive (fwdScore algebra dotPrimitive p pid m n) (fwdV algebra p pid m n) i=
      ∑ u : Fin p.N, if u.val/p.BT=n ∧ u.val ≤ n*p.BT+i.1.val then
        score p pid.head m (n*p.BT+i.1.val) u.val*value p pid.head m u.val (pid.channel*p.BM+i.2.1.val) else 0 := by
  rw [dot_eq]
  calc
    _ = ∑ u : Fin p.BT, if u.val ≤ i.1.val then
        (if n*p.BT+u.val<p.N then score p pid.head m (n*p.BT+i.1.val) (n*p.BT+u.val)*
          value p pid.head m (n*p.BT+u.val) (pid.channel*p.BM+i.2.1.val) else 0) else 0 := by
      apply Finset.sum_congr rfl
      intro u _
      by_cases hu : n*p.BT+u.val<p.N <;> by_cases hc : u.val ≤ i.1.val <;>
        simp only [Fin.le_def,fwdScore,dot_eq,fwdQ,fwdK,fwdV,read_eq,zero_eq,hu,ht,hc,ite_true,ite_false,
          mul_zero,zero_mul,Finset.sum_const_zero,score,query,key,value]
      all_goals simp [Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
    _ = _ := ChunkLinearAttn.Real.causal_block p.N p.NT p.BT n i.1.val hp hcover hn (fun u => score p pid.head m (n*p.BT+i.1.val) u*value p pid.head m u (pid.channel*p.BM+i.2.1.val))

theorem intra_query (p : Parameters) (hp : 0<p.BT) (hcover : p.N ≤ p.NT*p.BT)
    (head : Nat) (m : Memory ℝ) (n : Nat) (hn : n<p.NT)
    (i : TileIndex [p.BT,p.D]) (ht : n*p.BT+i.1.val<p.N) :
    intraDQ algebra dotPrimitive p head n m i=
      ∑ u : Fin p.N, if u.val/p.BT=n ∧ u.val ≤ n*p.BT+i.1.val then
        scoreGradient p head m (n*p.BT+i.1.val) u.val*key p head m u.val i.2.1.val else 0 := by
  unfold intraDQ
  rw [dot_eq]
  calc
    _ = ∑ u : Fin p.BT, if u.val ≤ i.1.val then
        (if n*p.BT+u.val<p.N then scoreGradient p head m (n*p.BT+i.1.val) (n*p.BT+u.val)*
          key p head m (n*p.BT+u.val) i.2.1.val else 0) else 0 := by
      apply Finset.sum_congr rfl
      intro u _
      by_cases hu : n*p.BT+u.val<p.N <;> by_cases hc : u.val ≤ i.1.val <;>
        simp only [intraDS,dot_eq,intraG,intraV,intraK,read_eq,zero_eq,hu,ht,hc,ite_true,ite_false,mul_zero,zero_mul,Finset.sum_const_zero,scoreGradient,incoming,value,key]
      all_goals simp [Nat.add_assoc]
    _ = _ := ChunkLinearAttn.Real.causal_block p.N p.NT p.BT n i.1.val hp hcover hn (fun u => scoreGradient p head m (n*p.BT+i.1.val) u*key p head m u i.2.1.val)

theorem intra_key (p : Parameters) (hp : 0<p.BT) (hcover : p.N ≤ p.NT*p.BT)
    (head : Nat) (m : Memory ℝ) (n : Nat) (hn : n<p.NT)
    (i : TileIndex [p.D,p.BT]) (hu : n*p.BT+i.2.1.val<p.N) :
    intraDKT algebra dotPrimitive p head n m i=
      ∑ t : Fin p.N, if t.val/p.BT=n ∧ n*p.BT+i.2.1.val ≤ t.val then
        query p head m t.val i.1.val*scoreGradient p head m t.val (n*p.BT+i.2.1.val) else 0 := by
  unfold intraDKT
  rw [dot_eq]
  calc
    _ = ∑ t : Fin p.BT, if i.2.1.val ≤ t.val then
        (if n*p.BT+t.val<p.N then query p head m (n*p.BT+t.val) i.1.val*
          scoreGradient p head m (n*p.BT+t.val) (n*p.BT+i.2.1.val) else 0) else 0 := by
      apply Finset.sum_congr rfl
      intro t _
      by_cases ht : n*p.BT+t.val<p.N <;> by_cases hc : i.2.1.val ≤ t.val <;>
        simp only [intraQ,intraDS,dot_eq,intraG,intraV,read_eq,zero_eq,hu,ht,hc,ite_true,ite_false,mul_zero,zero_mul,Finset.sum_const_zero,scoreGradient,query,incoming,value]
      all_goals simp [Nat.add_assoc]
    _ = _ := ChunkLinearAttn.Real.future_block p.N p.NT p.BT n i.2.1.val hp hcover hn (fun t => query p head m t i.1.val*scoreGradient p head m t (n*p.BT+i.2.1.val))

theorem intra_value (p : Parameters) (hp : 0<p.BT) (hcover : p.N ≤ p.NT*p.BT)
    (head : Nat) (m : Memory ℝ) (n : Nat) (hn : n<p.NT)
    (i : TileIndex [p.BT,p.E]) (hu : n*p.BT+i.1.val<p.N) :
    intraDV algebra dotPrimitive p head n m i=
      ∑ t : Fin p.N, if t.val/p.BT=n ∧ n*p.BT+i.1.val ≤ t.val then
        score p head m t.val (n*p.BT+i.1.val)*incoming p head m t.val i.2.1.val else 0 := by
  unfold intraDV
  rw [dot_eq]
  calc
    _ = ∑ t : Fin p.BT, if i.1.val ≤ t.val then
        (if n*p.BT+t.val<p.N then score p head m (n*p.BT+t.val) (n*p.BT+i.1.val)*
          incoming p head m (n*p.BT+t.val) i.2.1.val else 0) else 0 := by
      apply Finset.sum_congr rfl
      intro t _
      by_cases ht : n*p.BT+t.val<p.N <;> by_cases hc : i.1.val ≤ t.val <;>
        simp only [intraST,dot_eq,intraK,intraQ,intraG,read_eq,zero_eq,hu,ht,hc,ite_true,ite_false,mul_zero,zero_mul,Finset.sum_const_zero,score,query,key,incoming]
      all_goals simp [Nat.add_assoc,mul_comm]
    _ = _ := ChunkLinearAttn.Real.future_block p.N p.NT p.BT n i.1.val hp hcover hn (fun t => score p head m t (n*p.BT+i.1.val)*incoming p head m t i.2.1.val)

#axiomsClean forward_intra
#axiomsClean intra_query
#axiomsClean intra_key
#axiomsClean intra_value
end
end VeriTile.Bench.Repairs.LightningAttention.Real
