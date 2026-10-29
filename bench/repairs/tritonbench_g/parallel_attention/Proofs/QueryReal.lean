import bench.repairs.tritonbench_g.parallel_attention.Proofs.ForwardReal

namespace VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
open scoped BigOperators

/-- The source value-shard cotangent dot, before scale and score factors. -/
theorem partial_dot (p : Plans) (site : List Nat) (s : Shape) (x : TensorInputs ℝ)
    (bh shard t j : Nat) (ht : t < s.T) (hj : j < s.T) :
    (if shard=0 then
      dot realOps p site s.BV
        (fun v => realOps.cast (load realOps s.T s.V x.do_ bh t (shard*s.BV+v)))
        (fun v => load realOps s.T s.V x.v bh j (shard*s.BV+v)) + dzLoad realOps s x bh t
    else dot realOps p site s.BV
        (fun v => realOps.cast (load realOps s.T s.V x.do_ bh t (shard*s.BV+v)))
        (fun v => load realOps s.T s.V x.v bh j (shard*s.BV+v)))=
      partialCotangent s x bh shard t j := by
  have dots : dot realOps p site s.BV
      (fun v => realOps.cast (load realOps s.T s.V x.do_ bh t (shard*s.BV+v)))
      (fun v => load realOps s.T s.V x.v bh j (shard*s.BV+v))=
      ∑ v∈Finset.range s.BV,if shard*s.BV+v < s.V then
        x.do_ bh t (shard*s.BV+v)*x.v bh j (shard*s.BV+v) else 0 := by
    rw [dot_real]
    apply Finset.sum_congr rfl
    intro v hv
    by_cases h : shard*s.BV+v < s.V <;> simp [load,ht,hj,h]
  rw [dots]
  by_cases h : shard=0 <;> simp [partialCotangent,h,dzLoad,ht]

def queryTerm (a : Ops α) (p : Plans) (s : Shape) (x : TensorInputs α)
    (tile : Tile) (r j u : Nat) (diagonal : Bool) : α :=
  let t := tile.chunk*64+r
  let start := (if diagonal then tile.chunk*64 else 0)+j*32
  let ds0 := dot a p [1,0,tile.bh,tile.chunk,tile.value,r,j,u] s.BV
    (fun v => a.cast (load a s.T s.V x.do_ tile.bh t (tile.value*s.BV+v)))
    (fun v => load a s.T s.V x.v tile.bh (start+u) (tile.value*s.BV+v))
  let ds1 := if tile.value=0 then a.add ds0 (dzLoad a s x tile.bh t) else ds0
  let raw := dot a p [1,1,tile.bh,tile.chunk,tile.value,r,j,u] s.BK
    (queryLoad a s x tile 64 r)
    (fun k => load a s.T s.K x.k tile.bh (start+u) (tile.key*s.BK+k))
  let masked := diagonal && decide (r < j*32+u)
  let ds := if diagonal then a.mul (if masked then a.zero else ds1) (a.scale s.K s.useScale) else ds1
  let sc := if masked then a.zero else raw
  a.cast (a.mul (a.mul a.two ds) sc)

def queryProduct (s : Shape) (x : TensorInputs ℝ) (tile : Tile) (r e j : Nat) : ℝ :=
  if j ≤ tile.chunk*64+r then
    2 * partialCotangent s x tile.bh tile.value (tile.chunk*64+r) j *
      score s.K (selectedScale s.K s.useScale) (headInputs x tile.bh) (tile.chunk*64+r) j *
      x.k tile.bh j e
  else 0

theorem queryTerm_product (p : Plans) (s : Shape) (x : TensorInputs ℝ)
    (tile : Tile) (r e j u : Nat) (diagonal : Bool) (hk : tile.key=0)
    (fits : s.K ≤ s.BK) (ht : tile.chunk*64+r < s.T) (he : e < s.K) (hu : u < 32)
    (previous : diagonal=false → j < tile.chunk*2) :
    queryTerm realOps p s x tile r j u diagonal *
      load realOps s.T s.K x.k tile.bh
        ((if diagonal then tile.chunk*64 else 0)+j*32+u) (tile.key*s.BK+e)=
      (if diagonal then selectedScale s.K s.useScale else 1) *
        queryProduct s x tile r e ((if diagonal then tile.chunk*64 else 0)+j*32+u) := by
  have qeq : queryLoad realOps s x tile 64 r =
      (fun f => realOps.cast (realOps.mul
        (load realOps s.T s.K x.q tile.bh (tile.chunk*64+r) f)
        (realOps.scale s.K s.useScale))) := by
    funext f
    simp [queryLoad,hk]
  cases diagonal with
  | false =>
    have hj := previous rfl
    have before : j*32+u ≤ tile.chunk*64+r := by omega
    have valid : j*32+u < s.T := lt_of_le_of_lt before ht
    have sc := scaled_query_dot p [1,1,tile.bh,tile.chunk,tile.value,r,j,u]
      s x tile.bh (tile.chunk*64+r) (j*32+u) ht valid fits
    have ds := partial_dot p [1,0,tile.bh,tile.chunk,tile.value,r,j,u]
      s x tile.bh tile.value (tile.chunk*64+r) (j*32+u) ht valid
    simp only [real_cast,real_mul,real_scale] at sc ds
    simp only [queryTerm,Bool.false_eq_true,if_false,Bool.false_and,Bool.false_eq_true,
      Nat.zero_add,qeq,hk,Nat.zero_mul,Nat.zero_add,real_cast,real_mul,real_two,real_add,real_scale,sc,ds]
    simp [queryProduct,before,load,valid,he]
  | true =>
    by_cases masked : r < j*32+u
    · have after : ¬tile.chunk*64+j*32+u ≤ tile.chunk*64+r := by omega
      simp [queryTerm,masked,queryProduct,after]
    · have before : tile.chunk*64+j*32+u ≤ tile.chunk*64+r := by omega
      have valid : tile.chunk*64+j*32+u < s.T := lt_of_le_of_lt before ht
      have sc := scaled_query_dot p [1,1,tile.bh,tile.chunk,tile.value,r,j,u]
        s x tile.bh (tile.chunk*64+r) (tile.chunk*64+j*32+u) ht valid fits
      have ds := partial_dot p [1,0,tile.bh,tile.chunk,tile.value,r,j,u]
        s x tile.bh tile.value (tile.chunk*64+r) (tile.chunk*64+j*32+u) ht valid
      simp only [real_cast,real_mul,real_scale] at sc ds
      simp only [queryTerm,if_true,Bool.true_and,masked,decide_false,
        Bool.false_eq_true,if_false,qeq,hk,Nat.zero_mul,Nat.zero_add,
        real_cast,real_mul,real_two,real_add,real_scale,sc,ds]
      simp only [queryProduct,if_pos before,load,valid,he,and_self,if_true]
      ring

theorem queryStep_real (p : Plans) (s : Shape) (x : TensorInputs ℝ)
    (tile : Tile) (r e j : Nat) (diagonal : Bool) (carry : ℝ) (hk : tile.key=0)
    (fits : s.K ≤ s.BK) (ht : tile.chunk*64+r < s.T) (he : e < s.K)
    (previous : diagonal=false → j < tile.chunk*2) :
    queryStep realOps p s x tile r e diagonal j carry=carry+
      (if diagonal then selectedScale s.K s.useScale else 1)*
      ∑ u∈Finset.range 32,queryProduct s x tile r e
        ((if diagonal then tile.chunk*64 else 0)+j*32+u) := by
  change carry+dot realOps p [1,2,tile.bh,tile.chunk,tile.value,r,j,e] 32
    (fun u => queryTerm realOps p s x tile r j u diagonal)
    (fun u => load realOps s.T s.K x.k tile.bh
      ((if diagonal then tile.chunk*64 else 0)+j*32+u) (tile.key*s.BK+e))=_
  rw [dot_real,Finset.mul_sum]
  congr 1
  apply Finset.sum_congr rfl
  intro u hu
  exact queryTerm_product p s x tile r e j u diagonal hk fits ht he (Finset.mem_range.mp hu) previous

/-- Prefix and diagonal cover exactly the logical causal prefix. -/
theorem causal_prefix_sum (t m T : Nat) (f : Nat → ℝ) (hm : t < m) (hT : t < T) :
    (∑ j∈Finset.range m,if j ≤ t then f j else 0)=
      ∑ j∈Finset.range T,if j ≤ t then f j else 0 := by
  have h (n : Nat) (hn : t < n) :
      (∑ j∈Finset.range n,if j ≤ t then f j else 0)=∑ j∈Finset.range (t+1),f j := by
    simp_rw [← Nat.lt_succ_iff]
    exact sum_padded (t+1) n (by omega) f
  exact (h m hm).trans (h T hT).symm

theorem query_whole_real (p : Plans) (s : Shape) (x : TensorInputs ℝ)
    (tile : Tile) (r e : Nat) (hk : tile.key=0) (fits : s.K ≤ s.BK)
    (hr : r < 64) (ht : tile.chunk*64+r < s.T) (he : e < s.K) :
    queryLane realOps p s x tile r e=
      selectedScale s.K s.useScale * ∑ j∈Finset.range s.T,queryProduct s x tile r e j := by
  let f := queryProduct s x tile r e
  let scale := selectedScale s.K s.useScale
  have prior : foldN (queryStep realOps p s x tile r e false) (tile.chunk*2) 0=
      ∑ j∈Finset.range (tile.chunk*64),f j := by
    rw [foldN_bounded_congr _ (fun j carry => carry+∑ u∈Finset.range 32,f (j*32+u)) _ _
      (by intro j hj carry;simpa [f] using queryStep_real p s x tile r e j false carry hk fits ht he (by intro _;exact hj))]
    rw [foldN_add,sum_tiles]
    simp [show tile.chunk*2*32=tile.chunk*64 by omega]
  have diagonal : ∀ carry : ℝ,
      foldN (queryStep realOps p s x tile r e true) 2 carry=
      carry+scale*∑ j∈Finset.range 64,f (tile.chunk*64+j) := by
    intro carry
    rw [foldN_bounded_congr _ (fun j c => c+scale*∑ u∈Finset.range 32,f (tile.chunk*64+(j*32+u))) _ _
      (by intro j hj c;simpa [f,scale,Nat.add_assoc] using queryStep_real p s x tile r e j true c hk fits ht he (by simp))]
    rw [foldN_add,← Finset.mul_sum,sum_tiles 2 32 (fun j => f (tile.chunk*64+j))]
  unfold queryLane
  simp only [real_zero,real_mul,real_scale,real_cast]
  rw [prior,diagonal]
  change _*(_)+(scale*_) = _
  rw [mul_comm (∑ j∈Finset.range (tile.chunk*64),f j) _,← mul_add,← Finset.sum_range_add]
  congr 1
  exact causal_prefix_sum (tile.chunk*64+r) (tile.chunk*64+64) s.T _ (by omega) ht

#axiomsClean partial_dot
#axiomsClean queryTerm_product
#axiomsClean queryStep_real
#axiomsClean query_whole_real
end
end VeriTile.Bench.Repairs.ParallelAttention
