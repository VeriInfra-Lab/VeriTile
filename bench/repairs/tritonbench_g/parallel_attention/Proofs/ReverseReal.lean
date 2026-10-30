import bench.repairs.tritonbench_g.parallel_attention.Proofs.KeyValueReal

namespace VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
open scoped BigOperators

/-- Source descending blocks are a permutation of the bounded future suffix. -/
theorem reverse_blocks (f : Nat → ℝ) (M N S : Nat) (hNM : N ≤ M) :
    (∑ j∈Finset.range N,∑ u∈Finset.range S,f ((M-1-j)*S+u))=
      ∑ n∈Finset.range (N*S),f ((M-N)*S+n) := by
  calc
    _ = ∑ j∈Finset.range N,∑ u∈Finset.range S,f ((M-N)*S+((N-1-j)*S+u)) := by
      apply Finset.sum_congr rfl
      intro j hj
      have hj' := Finset.mem_range.mp hj
      have h : M-1-j=M-N+(N-1-j) := by omega
      simp [h,Nat.add_mul,Nat.add_assoc]
    _ = ∑ j∈Finset.range N,∑ u∈Finset.range S,f ((M-N)*S+(j*S+u)) :=
      Finset.sum_range_reflect (fun j => ∑ u∈Finset.range S,f ((M-N)*S+(j*S+u))) N
    _ = _ := sum_tiles N S (fun n => f ((M-N)*S+n))

theorem future_tail (f : Nat → ℝ) (M c : Nat) :
    (∑ n∈Finset.range ((M-(c+1)*2)*32),f ((M-(M-(c+1)*2))*32+n))=
      ∑ n∈Finset.range ((M-(c+1)*2)*32),f (c*64+64+n) := by
  by_cases h : (c+1)*2 ≤ M
  · have he : M-(M-(c+1)*2)=(c+1)*2 := by omega
    rw [he]
    congr 1
    funext n
    congr 1
    omega
  · have hz : M-(c+1)*2=0 := by omega
    simp [hz]

theorem suffix_coverage (f : Nat → ℝ) (T base tail : Nat)
    (bound : T ≤ base+64+tail)
    (before : ∀ t,t < base → f t=0) (after : ∀ t,T ≤ t → f t=0) :
    (∑ n∈Finset.range tail,f (base+64+n))+
      (∑ n∈Finset.range 64,f (base+n))=∑ t∈Finset.range T,f t := by
  have hprefix : (∑ t∈Finset.range base,f t)=0 :=
    Finset.sum_eq_zero (fun t ht => before t (Finset.mem_range.mp ht))
  have padded : (∑ t∈Finset.range (base+(64+tail)),f t)=∑ t∈Finset.range T,f t := by
    symm
    apply Finset.sum_subset (Finset.range_mono (by omega))
    intro t ht hn
    exact after t (by simpa using hn)
  rw [add_comm]
  simp_rw [Nat.add_assoc]
  rw [← Finset.sum_range_add]
  have join := Finset.sum_range_add f base (64+tail)
  rw [hprefix,zero_add] at join
  rw [← join,padded]

theorem source_suffix (s : Shape) (tile : Tile) (f : Nat → ℝ)
    (before : ∀ t,t < tile.chunk*64 → f t=0) (after : ∀ t,s.T ≤ t → f t=0) :
    (∑ j∈Finset.range (blocks s.T 32-(tile.chunk+1)*2),∑ u∈Finset.range 32,
        f ((blocks s.T 32-1-j)*32+u))+
      (∑ j∈Finset.range 2,∑ u∈Finset.range 32,f (tile.chunk*64+j*32+u))=
      ∑ t∈Finset.range s.T,f t := by
  rw [reverse_blocks _ _ _ _ (Nat.sub_le _ _),future_tail]
  have diag : (∑ j∈Finset.range 2,∑ u∈Finset.range 32,f (tile.chunk*64+j*32+u))=
      ∑ n∈Finset.range 64,f (tile.chunk*64+n) := by
    simpa [Nat.add_assoc] using sum_tiles 2 32 (fun n => f (tile.chunk*64+n))
  rw [diag]
  apply suffix_coverage _ _ _ _ _ before after
  have cover := blocks_cover s.T 32 (by norm_num)
  omega

/-- Both DK/DV loops, including descending future blocks and padded rows. -/
theorem keyValue_whole_real (p : Plans) (s : Shape) (x : TensorInputs ℝ)
    (tile : Tile) (r e v : Nat) (hk : tile.key=0) (fits : s.K ≤ s.BK)
    (hn : tile.chunk*64+r < s.T) (he : e < s.K)
    (hv : tile.value*s.BV+v < s.V) (hr : r < 64) :
    keyValueLane realOps p s x tile r e v=
      (∑ t∈Finset.range s.T,keyProduct s x tile r e t,
       ∑ t∈Finset.range s.T,valueProduct s x tile r v t) := by
  let f := keyProduct s x tile r e
  let g := valueProduct s x tile r v
  have phase (diagonal : Bool) (N : Nat) (carry : ℝ×ℝ)
      (hN : diagonal=false → N ≤ blocks s.T 32-(tile.chunk+1)*2) :
      foldN (keyValueStep realOps p s x tile r e v diagonal) N carry=
        (carry.1+∑ j∈Finset.range N,∑ u∈Finset.range 32,f (kvStart s tile diagonal j+u),
         carry.2+∑ j∈Finset.range N,∑ u∈Finset.range 32,g (kvStart s tile diagonal j+u)) := by
    rw [foldN_bounded_congr _ (fun j c =>
      (c.1+∑ u∈Finset.range 32,f (kvStart s tile diagonal j+u),
       c.2+∑ u∈Finset.range 32,g (kvStart s tile diagonal j+u))) _ _
      (by
        intro j hj c
        exact keyValueStep_real p s x tile r e v j diagonal c hk fits hn he hv hr
          (by intro hd;exact lt_of_lt_of_le hj (hN hd)))]
    exact foldN_pair_add _ _ _ _
  unfold keyValueLane
  simp only [real_zero]
  rw [phase false _ _ (by intro _;rfl),phase true _ _ (by simp)]
  simp only [real_cast,Prod.fst,Prod.snd,zero_add,kvStart,Bool.false_eq_true,if_false,if_true]
  apply Prod.ext <;> dsimp only
  · apply source_suffix
    · intro t ht
      have hc : ¬tile.chunk*64+r ≤ t := by omega
      simp [f,keyProduct,hc]
    · intro t ht
      simp [f,keyProduct,Nat.not_lt.mpr ht]
  · apply source_suffix
    · intro t ht
      have hc : ¬tile.chunk*64+r ≤ t := by omega
      simp [g,valueProduct,hc]
    · intro t ht
      simp [g,valueProduct,Nat.not_lt.mpr ht]

#axiomsClean reverse_blocks
#axiomsClean source_suffix
#axiomsClean keyValue_whole_real
end
end VeriTile.Bench.Repairs.ParallelAttention
