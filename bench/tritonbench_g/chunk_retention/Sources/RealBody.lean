import bench.tritonbench_g.chunk_retention.Sources.RealScalar
import bench.tritonbench_g.chunk_retention.Sources.ScalarOutput

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001
noncomputable section
open scoped BigOperators
namespace RealScalar

theorem sum_trim (f : Nat→ℝ) (small large : Nat) (h : small ≤ large)
    (zero : ∀ i, small  ≤  i → f i=0) :
    (∑ i∈Finset.range large,f i)=∑ i∈Finset.range small,f i := by
  symm
  apply Finset.sum_subset (Finset.range_mono h)
  intro i hi hn
  exact zero i (by simpa using hn)

theorem block_dot (width blocks size : Nat) (l r : Nat→ℝ)
    (h : size ≤ blocks*width) (zero : ∀ i, size  ≤  i → l i*r i=0) :
    blockChain realOps width l r blocks 0=∑ i∈Finset.range size,l i*r i := by
  rw [block_sum,zero_add,sum_trim _ size _ h zero]

theorem decay_succ (c : Shape) (h n : Nat) :
    decay realOps c h ((n:Int)+1)=(MathTarget.headRate c h)^(n+1) := by
  simpa using decay_nat c h (n+1)

theorem causal_real (c : Shape) (h r t : Nat) :
    causalWeight realOps c h r t=
      if t ≤ r then (MathTarget.headRate c h)^(r-t) else 0 := by
  unfold causalWeight
  split_ifs with ht
  · have he : (r:Int)-t=((r-t:Nat):Int) := by omega
    rw [he,decay_nat]
  · rfl

theorem score_real (c : Shape) (xs : Inputs ℝ) (h chunk r t : Nat)
    (hr : chunk*64+r<c.T) (ht : chunk*64+t<c.T) :
    Body.score realOps c xs h chunk r t=
      ∑ e∈Finset.range c.K,xs.q h (chunk*64+r) e*xs.k h (chunk*64+t) e := by
  rw [Body.score,real_zero,block_dot c.BK c.keyBlocks c.K _ _ c.key_cover]
  · apply Finset.sum_congr rfl
    intro e he
    simp [qAt,kAt,masked,hr,ht,Finset.mem_range.mp he]
  · intro e he
    simp [qAt,kAt,masked,Nat.not_lt.mpr he,real_zero]

theorem output_carry_real (c : Shape) (xs : Inputs ℝ) (h chunk r j : Nat)
    (history : Tile ℝ) (hr : chunk*64+r<c.T) (hj : j<c.V) :
    Body.outputCarry realOps c xs h chunk r j history=
      ∑ e∈Finset.range c.K,(xs.q h (chunk*64+r) e*
        (MathTarget.headRate c h)^(r+1))*history e j := by
  simp only [Body.outputCarry,real_mul,real_cast,real_zero,decay_succ]
  rw [block_dot c.BK c.keyBlocks c.K _ _ c.key_cover]
  · apply Finset.sum_congr rfl
    intro e he
    simp [qAt,masked,hr,hj,Finset.mem_range.mp he]
  · intro e he
    simp [qAt,masked,Nat.not_lt.mpr he,real_zero]

theorem output_real (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (h chunk r j : Nat)
    (history : Tile ℝ) (hr : chunk*64+r<c.T) (hr64 : r<64) (hj : j<c.V) :
    Body.output realOps c xs scale h chunk history r j=
      ScalarSource.outputTile c xs scale h chunk r j history := by
  simp only [Body.output,real_mul,real_cast,chain_sum,
    output_carry_real c xs h chunk r j history hr hj,ScalarSource.outputTile]
  congr 2
  rw [MathTarget.sum_prefix _ (r+1) 64 (by omega)]
  apply Finset.sum_congr rfl
  intro t ht
  by_cases htr : t ≤ r
  · have htT : chunk*64+t<c.T := by omega
    simp only [causal_real,if_pos htr,score_real c xs h chunk r t hr htT,
      vAt,masked,if_pos (And.intro htT hj),if_pos (show t<r+1 by omega)]
    ring
  · simp [causal_real,htr,show ¬t<r+1 by omega]


theorem ds_real (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (h chunk r t : Nat)
    (hr : chunk*64+r<c.T) (ht : chunk*64+t<c.T) :
    Body.ds realOps c xs scale h chunk r t=
      (∑ j∈Finset.range c.V,xs.incoming h (chunk*64+r) j*xs.v h (chunk*64+t) j)*
        ((if t ≤ r then (MathTarget.headRate c h)^(r-t) else 0)*scale) := by
  simp only [Body.ds,real_mul,real_cast,real_zero,causal_real]
  rw [block_dot c.BV c.valueBlocks c.V _ _ c.value_cover]
  · congr 1
    apply Finset.sum_congr rfl
    intro j hj
    simp [gAt,vAt,masked,hr,ht,Finset.mem_range.mp hj]
  · intro j hj
    simp [gAt,vAt,masked,Nat.not_lt.mpr hj,real_zero]

theorem ds_zero (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (h chunk r t : Nat)
    (ht : ¬t ≤ r) : Body.ds realOps c xs scale h chunk r t=0 := by
  simp [Body.ds,causal_real,ht,real_mul,real_cast]

theorem query_carry_real (c : Shape) (xs : Inputs ℝ) (h chunk r e : Nat)
    (history : Tile ℝ) (hr : chunk*64+r<c.T) (he : e<c.K) :
    blockChain realOps c.BV (fun j => gAt realOps c xs h (chunk*64+r) j)
      (fun j => masked realOps c.K c.V history e j) c.valueBlocks realOps.zero=
      ∑ j∈Finset.range c.V,xs.incoming h (chunk*64+r) j*history e j := by
  rw [real_zero,block_dot c.BV c.valueBlocks c.V _ _ c.value_cover]
  · apply Finset.sum_congr rfl
    intro j hj
    simp [gAt,masked,hr,he,Finset.mem_range.mp hj]
  · intro j hj
    simp [gAt,masked,Nat.not_lt.mpr hj,real_zero]

theorem query_real (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (h chunk r e : Nat)
    (history : Tile ℝ) (hr : chunk*64+r<c.T) (hr64 : r<64) (he : e<c.K) :
    Body.dq realOps c xs scale h chunk history r e=
      ScalarSource.queryTile c xs scale h chunk r e history := by
  simp only [Body.dq,chain_sum,real_cast,real_mul,decay_succ,
    query_carry_real c xs h chunk r e history hr he,ScalarSource.queryTile]
  rw [sum_trim _ (r+1) 64 (by omega) (by
    intro t ht
    rw [ds_zero c xs scale h chunk r t (by omega),zero_mul])]
  rw [add_mul]
  congr 1
  · rw [Finset.sum_mul,Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro j hj
    ring
  · rw [Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro t ht
    have htr : t ≤ r := by have := Finset.mem_range.mp ht; omega
    have htT : chunk*64+t<c.T := by omega
    rw [ds_real c xs scale h chunk r t hr htT]
    simp only [if_pos htr,kAt,masked,if_pos (And.intro htT he)]
    ring


theorem sum_suffix (f : Nat→ℝ) (start size total : Nat) (hs : start ≤ size) (ht : size ≤ total)
    (before : ∀ i, i<start → f i=0) (after : ∀ i, size ≤ i → i<total → f i=0) :
    (∑ i∈Finset.range total,f i)=∑ i∈Finset.range (size-start),f (start+i) := by
  have trim : (∑ i∈Finset.range size,f i)=∑ i∈Finset.range total,f i := by
    apply Finset.sum_subset (Finset.range_mono ht)
    intro i hi hn
    exact after i (by simpa using hn) (Finset.mem_range.mp hi)
  rw [←trim,MathTarget.sum_shift f start size hs]
  apply Finset.sum_congr rfl
  intro i hi
  split_ifs with hb
  · rfl
  · exact before i (by omega)

theorem key_carry_real (c : Shape) (xs : Inputs ℝ) (h chunk t e : Nat)
    (history : Tile ℝ) (ht : chunk*64+t<c.T) (he : e<c.K) :
    blockChain realOps c.BV (fun j => vAt realOps c xs h (chunk*64+t) j)
      (fun j => masked realOps c.K c.V history e j) c.valueBlocks realOps.zero=
      ∑ j∈Finset.range c.V,xs.v h (chunk*64+t) j*history e j := by
  rw [real_zero,block_dot c.BV c.valueBlocks c.V _ _ c.value_cover]
  · apply Finset.sum_congr rfl
    intro j hj
    simp [vAt,masked,ht,he,Finset.mem_range.mp hj]
  · intro j hj
    simp [vAt,masked,Nat.not_lt.mpr hj,real_zero]

theorem key_real (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (h chunk t e : Nat)
    (history : Tile ℝ) (hc : chunk<c.chunks) (ht : t<c.length chunk) (he : e<c.K) :
    Body.dk realOps c xs scale h chunk history t e=
      ScalarSource.keyTile c xs scale h chunk t e history := by
  have ht64 : t<64 := lt_of_lt_of_le ht (c.length_le chunk)
  have htT := (time_lane c chunk t hc ht64).mpr ht
  have de : (c.length chunk:Int)-t-1=((c.length chunk-1-t:Nat):Int) := by omega
  have carry := key_carry_real c xs h chunk t e history htT he
  simp only [real_zero] at carry
  simp only [Body.dk,real_mul,real_add,real_cast,real_zero,chain_sum,zero_add,
    carry,de,decay_nat,ScalarSource.keyTile]
  congr 1
  rw [sum_suffix _ t (c.length chunk) 64 (by omega) (c.length_le chunk) (by
    intro r hr
    rw [ds_zero c xs scale h chunk r t (by omega),mul_zero]) (by
    intro r hr hr64
    have hn : ¬chunk*64+r<c.T := fun H=>(Nat.not_lt.mpr hr) ((time_lane c chunk r hc hr64).mp H)
    simp [qAt,masked,hn,real_zero])]
  apply Finset.sum_congr rfl
  intro i hi
  have hr : t+i<c.length chunk := by have := Finset.mem_range.mp hi; omega
  have hr64 : t+i<64 := lt_of_lt_of_le hr (c.length_le chunk)
  have hrT := (time_lane c chunk (t+i) hc hr64).mpr hr
  rw [ds_real c xs scale h chunk (t+i) t hrT htT]
  simp only [qAt,masked,if_pos (And.intro hrT he),if_pos (show t ≤ t+i by omega),Nat.add_sub_cancel_left]
  have dot : (∑ j∈Finset.range c.V,xs.incoming h (chunk*64+(t+i)) j*xs.v h (chunk*64+t) j)=
      ∑ j∈Finset.range c.V,xs.v h (chunk*64+t) j*xs.incoming h (chunk*64+t+i) j := by
    apply Finset.sum_congr rfl
    intro j hj
    rw [Nat.add_assoc]
    ring
  rw [dot]
  rw [←Nat.add_assoc]
  ring


def keyVector (c : Shape) (xs : Inputs ℝ) (block h chunk t i : Nat) : ℝ :=
  if block*c.BK+i<c.K then xs.k h (chunk*64+t) (block*c.BK+i) else 0

theorem partial_score_real (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (block h chunk t r : Nat)
    (ht : chunk*64+t<c.T) (hr : chunk*64+r<c.T) :
    Body.partialScore realOps c xs scale block h chunk t r=
      (∑ i∈Finset.range c.BK,keyVector c xs block h chunk t i*
        xs.q h (chunk*64+r) (block*c.BK+i))*
      ((if t ≤ r then (MathTarget.headRate c h)^(r-t) else 0)*scale) := by
  simp only [Body.partialScore,real_mul,chain_sum,real_zero,zero_add,causal_real]
  congr 1
  apply Finset.sum_congr rfl
  intro i hi
  by_cases he : block*c.BK+i<c.K
  · simp [keyVector,kAt,qAt,masked,ht,hr,he]
  · simp [keyVector,kAt,qAt,masked,he,real_zero]

theorem partial_score_zero (c : Shape) (xs : Inputs ℝ) (scale : ℝ)
    (block h chunk t r : Nat) (ht : ¬t ≤ r) :
    Body.partialScore realOps c xs scale block h chunk t r=0 := by
  simp [Body.partialScore,causal_real,ht,real_mul]

theorem value_carry_real (c : Shape) (xs : Inputs ℝ) (block h chunk t j : Nat)
    (history : Tile ℝ) (ht : chunk*64+t<c.T) (hj : j<c.V) :
    chain realOps c.BK (fun i => kAt realOps c xs h (chunk*64+t) (block*c.BK+i))
      (fun i => masked realOps c.K c.V history (block*c.BK+i) j) realOps.zero=
      ∑ i∈Finset.range c.BK,keyVector c xs block h chunk t i*history (block*c.BK+i) j := by
  rw [chain_sum,real_zero,zero_add]
  apply Finset.sum_congr rfl
  intro i hi
  by_cases he : block*c.BK+i<c.K
  · simp [keyVector,kAt,masked,ht,hj,he]
  · simp [keyVector,kAt,masked,he,real_zero]

theorem value_real (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (block h chunk t j : Nat)
    (history : Tile ℝ) (hc : chunk<c.chunks) (ht : t<c.length chunk) (hj : j<c.V) :
    Body.dvPartial realOps c xs scale block h chunk history t j=
      ScalarSource.valuePartial c xs scale block h chunk t j history := by
  have ht64 : t<64 := lt_of_lt_of_le ht (c.length_le chunk)
  have htT := (time_lane c chunk t hc ht64).mpr ht
  have de : (c.length chunk:Int)-t-1=((c.length chunk-1-t:Nat):Int) := by omega
  have carry := value_carry_real c xs block h chunk t j history htT hj
  simp only [chain_sum,real_zero,zero_add] at carry
  simp only [Body.dvPartial,real_cast,real_mul,chain_sum,real_zero,zero_add,carry,de,decay_nat]
  rw [sum_suffix _ t (c.length chunk) 64 (by omega) (c.length_le chunk) (by
    intro r hr
    rw [partial_score_zero c xs scale block h chunk t r (by omega),zero_mul]) (by
    intro r hr hr64
    have hn : ¬chunk*64+r<c.T := fun H=>(Nat.not_lt.mpr hr) ((time_lane c chunk r hc hr64).mp H)
    simp [gAt,masked,hn,real_zero])]
  have localSum :
      (∑ i∈Finset.range (c.length chunk-t),
        Body.partialScore realOps c xs scale block h chunk t (t+i)*gAt realOps c xs h (chunk*64+(t+i)) j)=
      ∑ i∈Finset.range (c.length chunk-t),
        (∑ e∈Finset.range c.BK,keyVector c xs block h chunk t e*xs.q h (chunk*64+t+i) (block*c.BK+e))*
          ((MathTarget.headRate c h)^i*scale*xs.incoming h (chunk*64+t+i) j) := by
    apply Finset.sum_congr rfl
    intro i hi
    have hr : t+i<c.length chunk := by have := Finset.mem_range.mp hi; omega
    have hrT := (time_lane c chunk (t+i) hc (lt_of_lt_of_le hr (c.length_le chunk))).mpr hr
    rw [partial_score_real c xs scale block h chunk t (t+i) htT hrT]
    have hrT2 : chunk*64+t+i<c.T := by omega
    simp only [if_pos (show t ≤ t+i by omega),Nat.add_sub_cancel_left,
      gAt,masked,if_pos (And.intro hrT hj),if_pos (And.intro hrT2 hj),←Nat.add_assoc]
    ring
  rw [localSum,ScalarAlgebra.contraction]
  unfold ScalarSource.valuePartial ScalarSource.localCoefficient MathTarget.forcing
  apply Finset.sum_congr rfl
  intro e he
  by_cases hk : block*c.BK+e<c.K
  · simp only [keyVector,if_pos hk]
    congr 1
    congr 1
    apply Finset.sum_congr rfl
    intro i hi
    ring
  · simp [keyVector,hk]

#axiomsClean output_real
#axiomsClean query_real
#axiomsClean key_real
#axiomsClean value_real
end RealScalar
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001
