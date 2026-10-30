import bench.tritonbench_g.parallel_retention_attention.Proofs.Arithmetic

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention
noncomputable section
open scoped BigOperators

/-- The host's nonnegative ceiling division. -/
def blocks (n b : Nat) := (n+b-1)/b

theorem blocks_cover (n b : Nat) (hb : 0<b) : n≤blocks n b*b := by
  have hmod := Nat.mod_lt (n+b-1) hb
  have hdiv := Nat.div_add_mod (n+b-1) b
  unfold blocks
  nlinarith [Nat.sub_add_cancel (show 1≤n+b by omega)]

theorem tile_owner (n b i : Nat) (hb : 0<b) (hi : i<n) :
    i/b<blocks n b ∧ i%b<b ∧ i/b*b+i%b=i := by
  have hc := blocks_cover n b hb
  refine ⟨?_,Nat.mod_lt i hb,?_⟩
  · exact (Nat.div_lt_iff_lt_mul hb).2 (lt_of_lt_of_le hi hc)
  · simpa [Nat.mul_comm] using Nat.div_add_mod i b

theorem tile_injective (b c d i j : Nat) (hi : i<b) (hj : j<b)
    (eq : c*b+i=d*b+j) : c=d ∧ i=j := by
  have hc : c = d := by
    rcases lt_trichotomy c d with hlt | heq | hgt
    · have hstep : c+1 ≤ d := by omega
      nlinarith
    · exact heq
    · have hstep : d+1 ≤ c := by omega
      nlinarith
  exact ⟨hc, by simpa [hc] using eq⟩

theorem sum_tiles (n b : Nat) (f : Nat → ℝ) :
    (∑ c∈Finset.range n,∑ i∈Finset.range b,f (c*b+i))=
      ∑ i∈Finset.range (n*b),f i := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Finset.sum_range_succ,ih,Nat.succ_mul,Finset.sum_range_add]

theorem sum_padded (n m : Nat) (h : n ≤ m) (f : Nat → ℝ) :
    (∑ i∈Finset.range m,if i<n then f i else 0)=∑ i∈Finset.range n,f i := by
  have e : (∑ i∈Finset.range n,if i<n then f i else 0)=∑ i∈Finset.range n,f i := by
    apply Finset.sum_congr rfl
    intro i hi
    simp only [Finset.mem_range] at hi
    simp [hi]
  rw [← e]
  exact (Finset.sum_subset (Finset.range_mono h) (by
    intro i hi hn
    simp only [Finset.mem_range,not_lt] at hn
    simp [Nat.not_lt.mpr hn])).symm

theorem tiled_partition (n b : Nat) (hb : 0<b) (f : Nat → ℝ) :
    (∑ c∈Finset.range (blocks n b),∑ i∈Finset.range b,
       if c*b+i<n then f (c*b+i) else 0)=∑ i∈Finset.range n,f i := by
  rw [sum_tiles (blocks n b) b (fun i => if i<n then f i else 0)]
  exact sum_padded n _ (blocks_cover n b hb) f

/-- Source tensor layout after its existing contiguous decorator. Storage offsets
are part of the base pointer, never assumed to be zero. -/
def dense (T D bh t e : Nat) := (bh*T+t)*D+e

theorem dense_bound (T D M bh t e : Nat) (hb : bh<M) (ht : t<T) (he : e<D) :
    dense T D bh t e<M*T*D := by
  unfold dense
  have h : bh*T+t<M*T := by nlinarith
  nlinarith

theorem dense_injective (T D bh bh' t t' e e' : Nat)
    (ht : t<T) (ht' : t'<T) (he : e<D) (he' : e'<D)
    (h : dense T D bh t e=dense T D bh' t' e') :
    bh=bh' ∧ t=t' ∧ e=e' := by
  obtain ⟨hx,heq⟩ := tile_injective D (bh*T+t) (bh'*T+t') e e' he he' h
  obtain ⟨hb,htq⟩ := tile_injective T bh bh' t t' ht ht' hx
  exact ⟨hb,htq,heq⟩

/-- Bounds include inactive formed addresses and one-past pointer advances.
All block-pointer coordinates use signed 32-bit, other arithmetic signed 64-bit. -/
structure IndexDomain (B H T DK DV BK BV : Nat) : Prop where
  dk_pos : 0<DK
  dv_pos : 0<DV
  bk_pos : 0<BK
  bv_pos : 0<BV
  bk_tile : BK=16 ∨ BK=32 ∨ BK=64 ∨ BK=128
  bv_tile : BV=16 ∨ BV=32 ∨ BV=64 ∨ BV=128
  time_padded : blocks T 128*128+128<2^31
  key_padded : blocks DK BK*BK+BK<2^31
  value_padded : blocks DV BV*BV+BV<2^31
  grid_x : blocks DK BK*blocks DV BV<2^31
  grid_y : blocks T 64<65536
  grid_z : B*H<65536
  formed_key : ((blocks DV BV*B*H+1)*T+blocks T 128*128+128)*DK+
    blocks DK BK*BK+BK<2^61
  formed_value : ((blocks DK BK*B*H+1)*T+blocks T 128*128+128)*DV+
    blocks DV BV*BV+BV<2^61

#axiomsClean blocks_cover
#axiomsClean tile_owner
#axiomsClean tile_injective
#axiomsClean tiled_partition
#axiomsClean dense_bound
#axiomsClean dense_injective
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention
