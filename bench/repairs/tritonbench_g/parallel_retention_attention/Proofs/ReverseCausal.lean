import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.Future

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention
noncomputable section
open scoped BigOperators

theorem suffix_split (g : ℝ) (x : Nat → ℝ) (base L r tail T : Nat)
    (hr : r<L) (bound : T≤base+L+tail) (zero : ∀ n,T≤n → x n=0) :
    (∑ n∈Finset.range L,if r≤n then g^(n-r)*x (base+n) else 0)+
      (∑ n∈Finset.range tail,g^(L+n-r)*x (base+L+n))=
      ∑ t∈Finset.range T,if base+r≤t then g^(t-(base+r))*x t else 0 := by
  let f := fun t => if base+r≤t then g^(t-(base+r))*x t else 0
  have diag : (∑ n∈Finset.range L,if r≤n then g^(n-r)*x (base+n) else 0)=
      ∑ n∈Finset.range L,f (base+n) := by
    apply Finset.sum_congr rfl
    intro n hn
    simp only [f,Nat.add_le_add_iff_left,Nat.add_sub_add_left]
  have later : (∑ n∈Finset.range tail,g^(L+n-r)*x (base+L+n))=
      ∑ n∈Finset.range tail,f (base+(L+n)) := by
    apply Finset.sum_congr rfl
    intro n hn
    have hc : base+r≤base+(L+n) := by omega
    simp only [f,if_pos hc,Nat.add_sub_add_left,Nat.add_assoc]
  have hprefix : (∑ n∈Finset.range base,f n)=0 := by
    apply Finset.sum_eq_zero
    intro n hn
    have hc : ¬base+r≤n := by have := Finset.mem_range.mp hn;omega
    simp [f,hc]
  have padded : (∑ t∈Finset.range (base+(L+tail)),f t)=∑ t∈Finset.range T,f t := by
    symm
    apply Finset.sum_subset (Finset.range_mono (by omega))
    intro t ht hn
    have hT : T≤t := by simpa using hn
    simp [f,zero t hT]
  rw [diag,later,← Finset.sum_range_add]
  have join := Finset.sum_range_add f base (L+tail)
  rw [hprefix,zero_add] at join
  rw [← join,padded]

theorem future_tail (g : ℝ) (x : Nat → ℝ) (M c r : Nat) (hr : r<64) :
    (∑ n∈Finset.range ((M-(c+1)*2)*32),g^(n+(64-r))*x ((M-(M-(c+1)*2))*32+n))=
      ∑ n∈Finset.range ((M-(c+1)*2)*32),g^(64+n-r)*x (c*64+64+n) := by
  by_cases hm : (c+1)*2≤M
  · have he : M-(M-(c+1)*2)=(c+1)*2 := by omega
    rw [he]
    apply Finset.sum_congr rfl
    intro n hn
    have he1 : (c+1)*2*32+n=c*64+64+n := by omega
    have he2 : n+(64-r)=64+n-r := by omega
    rw [he1,he2]
  · have he : M-(c+1)*2=0 := by omega
    simp [he]

theorem padded_future_bound (T c : Nat) :
    T≤c*64+64+(blocks T 32-(c+1)*2)*32 := by
  have cover := blocks_cover T 32 (by norm_num)
  omega

theorem keyProduct_padding (s : Shape) (x : Inputs ℝ) (tile : Tile) (r e t : Nat)
    (ht : s.T≤t) : keyProduct s x tile r e t=0 := by
  unfold keyProduct
  simp [load,Nat.not_lt.mpr ht,real_zero]

theorem valueProduct_padding (s : Shape) (x : Inputs ℝ) (tile : Tile) (r v t : Nat)
    (ht : s.T≤t) : valueProduct s x tile r v t=0 := by
  unfold valueProduct
  simp [load,Nat.not_lt.mpr ht,real_zero]

theorem keyValue_causal (p : Plans) (s : Shape) (x : Inputs ℝ) (tile : Tile)
    (r e v : Nat) (hr : r<64) :
    keyValueLane realOps p s x tile r e v=
      ((Real.sqrt (s.DK:ℝ))⁻¹*(∑ t∈Finset.range s.T,if tile.chunk*64+r≤t then
        gamma (tile.bh%s.H)^(t-(tile.chunk*64+r))*keyProduct s x tile r e t else 0),
       (Real.sqrt (s.DK:ℝ))⁻¹*(∑ t∈Finset.range s.T,if tile.chunk*64+r≤t then
        gamma (tile.bh%s.H)^(t-(tile.chunk*64+r))*valueProduct s x tile r v t else 0)) := by
  rw [keyValue_phase_sums]
  apply Prod.ext <;> simp only [Prod.fst,Prod.snd]
  · congr 1
    unfold futureKey diagonalKey
    rw [future_tail _ _ _ _ _ hr,add_comm]
    exact suffix_split _ _ _ _ _ _ _ hr (padded_future_bound s.T tile.chunk)
      (keyProduct_padding s x tile r e)
  · congr 1
    unfold futureValue diagonalValue
    rw [future_tail _ _ _ _ _ hr,add_comm]
    exact suffix_split _ _ _ _ _ _ _ hr (padded_future_bound s.T tile.chunk)
      (valueProduct_padding s x tile r v)

#axiomsClean suffix_split
#axiomsClean future_tail
#axiomsClean padded_future_bound
#axiomsClean keyValue_causal
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention
