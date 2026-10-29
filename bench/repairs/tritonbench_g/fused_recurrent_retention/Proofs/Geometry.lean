import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.Arithmetic
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
noncomputable section
open scoped BigOperators

theorem block_positive (s : Shape) (h : s.Admitted) : 0<s.BK ∧ 0<s.BV := by
  exact ⟨lt_min h.1 (by decide),lt_min h.2.1 (by decide)⟩

theorem tileCount_source (n w : Nat) (hn : 0<n) (hw : 0<w) :
    tileCount n w=(n+w-1)/w := by
  unfold tileCount
  rw [show n+w-1=(n-1)+w by omega,Nat.add_div_right _ hw]

theorem tile_coverage (n w : Nat) (hn : 0<n) (hw : 0<w) :
    n≤tileCount n w*w := by
  have hmod := Nat.mod_lt (n-1) hw
  have he := Nat.mod_add_div (n-1) w
  have hn' : n-1+1=n := by omega
  unfold tileCount
  nlinarith

theorem owner_bound (n w i : Nat) (hi : i<n) : i/w<tileCount n w := by
  have he : i≤n-1 := by omega
  have hd : i/w≤(n-1)/w := Nat.div_le_div_right he
  unfold tileCount
  omega

theorem owner_reconstruct (w i : Nat) : (i/w)*w+i%w=i := by
  simpa only [Nat.mul_comm,Nat.add_comm] using Nat.mod_add_div i w

theorem pair_index_injective (n a b a' b' : Nat) (hb : b<n) (hb' : b'<n)
    (he : a*n+b=a'*n+b') : a=a' ∧ b=b' := by
  have ha : a=a' := by
    rcases lt_trichotomy a a' with h|h|h
    · have hm := Nat.mul_le_mul_right n (Nat.succ_le_of_lt h)
      nlinarith
    · exact h
    · have hm := Nat.mul_le_mul_right n (Nat.succ_le_of_lt h)
      nlinarith
  exact ⟨ha,by simpa only [ha,Nat.add_left_cancel_iff] using he⟩

theorem partialOffset_injective (s : Shape) (width split head time channel split' head' time' channel' : Nat)
    (hh : head<s.BH) (hh' : head'<s.BH) (ht : time<s.T) (ht' : time'<s.T)
    (hc : channel<width) (hc' : channel'<width)
    (he : partialOffset s width split head time channel=partialOffset s width split' head' time' channel') :
    split=split' ∧ head=head' ∧ time=time' ∧ channel=channel' := by
  obtain ⟨he,hc⟩ := pair_index_injective width _ _ _ _ hc hc' he
  obtain ⟨he,ht⟩ := pair_index_injective s.T _ _ _ _ ht ht' he
  obtain ⟨hs,hh⟩ := pair_index_injective s.BH _ _ _ _ hh hh' he
  exact ⟨hs,hh,ht,hc⟩

theorem stateOffset_injective (s : Shape) (h k v h' k' v' : Nat)
    (hk : k<s.DK) (hk' : k'<s.DK) (hv : v<s.DV) (hv' : v'<s.DV)
    (he : stateOffset s h k v=stateOffset s h' k' v') : h=h' ∧ k=k' ∧ v=v' := by
  obtain ⟨he,hv⟩ := pair_index_injective s.DV _ _ _ _ hv hv' he
  obtain ⟨hh,hk⟩ := pair_index_injective s.DK _ _ _ _ hk hk' he
  exact ⟨hh,hk,hv⟩

theorem sum_chunks (count width : Nat) (f : Nat → ℝ) :
    (∑ b∈Finset.range count,∑ j∈Finset.range width,f (b*width+j))=
      ∑ i∈Finset.range (count*width),f i := by
  induction count with
  | zero => simp
  | succ n ih => rw [Finset.sum_range_succ,ih,Nat.succ_mul,Finset.sum_range_add]

theorem sum_padded (n total : Nat) (hn : n≤total) (f : Nat → ℝ) :
    (∑ i∈Finset.range total,if i<n then f i else 0)=∑ i∈Finset.range n,f i := by
  have he : total=n+(total-n) := by omega
  rw [he,Finset.sum_range_add]
  have hz : (∑ i∈Finset.range (total-n),if n+i<n then f (n+i) else 0)=0 := by
    apply Finset.sum_eq_zero
    intro i hi
    simp
  rw [hz,add_zero]
  apply Finset.sum_congr rfl
  intro i hi
  simp [Finset.mem_range.mp hi]

theorem sum_tiles (n w : Nat) (hn : 0<n) (hw : 0<w) (f : Nat → ℝ) :
    (∑ b∈Finset.range (tileCount n w),∑ j∈Finset.range w,
      if b*w+j<n then f (b*w+j) else 0)=∑ i∈Finset.range n,f i := by
  exact (sum_chunks (tileCount n w) w (fun i => if i<n then f i else 0)).trans
    (sum_padded n _ (tile_coverage n w hn hw) f)

#axiomsClean tileCount_source
#axiomsClean tile_coverage
#axiomsClean owner_bound
#axiomsClean partialOffset_injective
#axiomsClean stateOffset_injective
#axiomsClean sum_tiles
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention
