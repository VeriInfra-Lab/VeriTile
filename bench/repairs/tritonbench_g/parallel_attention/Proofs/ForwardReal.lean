import bench.repairs.tritonbench_g.parallel_attention.Proofs.GraphReal

namespace VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
open scoped BigOperators

/-- The source score expression, factored only to state its local proof. -/
def forwardScore (a : Ops α) (p : Plans) (s : Shape) (x : TensorInputs α)
    (tile : Tile) (r j u : Nat) (diagonal : Bool) : α :=
  let start := (if diagonal then tile.chunk*128 else 0)+j*32
  let raw := dot a p [0,0,tile.bh,tile.chunk,tile.value,r,j,u] s.BK
    (queryLoad a s x tile 128 r)
    (fun e => load a s.T s.K x.k tile.bh (start+u) (tile.key*s.BK+e))
  if diagonal && decide (r<j*32+u) then a.zero else a.mul raw raw

theorem forwardScore_real (p : Plans) (s : Shape) (x : TensorInputs ℝ)
    (tile : Tile) (r j u : Nat) (diagonal : Bool) (hk : tile.key=0)
    (fits : s.K ≤ s.BK) (ht : tile.chunk*128+r<s.T) (hu : u<32)
    (previous : diagonal=false → j<tile.chunk*4) :
    forwardScore realOps p s x tile r j u diagonal=
      weight s.K (selectedScale s.K s.useScale) (headInputs x tile.bh)
        (tile.chunk*128+r) ((if diagonal then tile.chunk*128 else 0)+j*32+u) := by
  have qeq : queryLoad realOps s x tile 128 r =
      (fun e => realOps.cast (realOps.mul
        (load realOps s.T s.K x.q tile.bh (tile.chunk*128+r) e)
        (realOps.scale s.K s.useScale))) := by
    funext e
    simp [queryLoad,hk]
  cases diagonal with
  | false =>
    have hj := previous rfl
    have before : j*32+u ≤ tile.chunk*128+r := by nlinarith
    have valid : j*32+u<s.T := lt_of_le_of_lt before ht
    have score_eq := scaled_query_dot p [0,0,tile.bh,tile.chunk,tile.value,r,j,u]
      s x tile.bh (tile.chunk*128+r) (j*32+u) ht valid fits
    simpa [forwardScore,qeq,hk,weight,before,pow_two] using
      congrArg (fun v : ℝ => v*v) score_eq
  | true =>
    by_cases masked : r<j*32+u
    · have after : ¬tile.chunk*128+j*32+u ≤ tile.chunk*128+r := by omega
      simp [forwardScore,masked,weight,after]
    · have before : tile.chunk*128+j*32+u ≤ tile.chunk*128+r := by omega
      have valid : tile.chunk*128+j*32+u<s.T := lt_of_le_of_lt before ht
      have score_eq := scaled_query_dot p [0,0,tile.bh,tile.chunk,tile.value,r,j,u]
        s x tile.bh (tile.chunk*128+r) (tile.chunk*128+j*32+u) ht valid fits
      simpa [forwardScore,qeq,hk,weight,before,masked,pow_two] using
        congrArg (fun v : ℝ => v*v) score_eq

theorem weighted_load (s : Shape) (x : TensorInputs ℝ) (bh t j v : Nat)
    (ht : t<s.T) (hv : v<s.V) :
    weight s.K (selectedScale s.K s.useScale) (headInputs x bh) t j *
      load realOps s.T s.V x.v bh j v =
    weight s.K (selectedScale s.K s.useScale) (headInputs x bh) t j * x.v bh j v := by
  by_cases causal : j≤t
  · have hj : j<s.T := lt_of_le_of_lt causal ht
    simp [load,hj,hv]
  · simp [weight,causal]

theorem forwardStep_real (p : Plans) (s : Shape) (x : TensorInputs ℝ)
    (tile : Tile) (r v j : Nat) (diagonal : Bool) (carry : ℝ×ℝ)
    (hk : tile.key=0) (fits : s.K ≤ s.BK) (ht : tile.chunk*128+r<s.T)
    (hv : tile.value*s.BV+v<s.V) (previous : diagonal=false → j<tile.chunk*4) :
    forwardStep realOps p s x tile r v diagonal j carry=
      (carry.1+∑ u∈Finset.range 32,
        weight s.K (selectedScale s.K s.useScale) (headInputs x tile.bh)
          (tile.chunk*128+r) ((if diagonal then tile.chunk*128 else 0)+j*32+u) *
        x.v tile.bh ((if diagonal then tile.chunk*128 else 0)+j*32+u) (tile.value*s.BV+v),
       carry.2+∑ u∈Finset.range 32,
        weight s.K (selectedScale s.K s.useScale) (headInputs x tile.bh)
          (tile.chunk*128+r) ((if diagonal then tile.chunk*128 else 0)+j*32+u)) := by
  change (carry.1+dot realOps p [0,2,tile.bh,tile.chunk,tile.value,r,j,v] 32
      (fun u => forwardScore realOps p s x tile r j u diagonal)
      (fun u => load realOps s.T s.V x.v tile.bh
        ((if diagonal then tile.chunk*128 else 0)+j*32+u) (tile.value*s.BV+v)),
    carry.2+total realOps p [0,1,tile.bh,tile.chunk,tile.value,r,j] 32
      (fun u => forwardScore realOps p s x tile r j u diagonal))=_
  rw [dot_real,total_real]
  apply Prod.ext <;> dsimp only
  · congr 1
    apply Finset.sum_congr rfl
    intro u hu
    rw [forwardScore_real p s x tile r j u diagonal hk fits ht (Finset.mem_range.mp hu) previous]
    exact weighted_load s x tile.bh _ _ _ ht hv
  · congr 1
    apply Finset.sum_congr rfl
    intro u hu
    exact forwardScore_real p s x tile r j u diagonal hk fits ht (Finset.mem_range.mp hu) previous

theorem foldN_bounded_congr (f g : Nat → β → β) (n : Nat) (z : β)
    (same : ∀ i, i<n → ∀ carry,f i carry=g i carry) : foldN f n z=foldN g n z := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [foldN,foldN,same n (by omega),ih (by intro i hi;exact same i (by omega))]

theorem causal_sum_prefix (K t m : Nat) (scale : ℝ) (x : Inputs ℝ) (f : Nat → ℝ)
    (bound : t + 1 ≤ m) :
    (∑ j∈Finset.range m,weight K scale x t j*f j)=
      ∑ j∈Finset.range (t+1),(score K scale x t j)^2*f j := by
  have terms : (fun j => weight K scale x t j*f j)=
      (fun j => if j<t+1 then (score K scale x t j)^2*f j else 0) := by
    funext j
    by_cases h : j≤t <;> simp [weight,h,Nat.lt_succ_iff]
  rw [terms]
  exact sum_padded (t+1) m bound _

/-- Complete forward prefix and all four diagonal blocks connect to the
independent logical O/Z formula, for every active output lane and ragged tile. -/
theorem forward_whole_real (p : Plans) (s : Shape) (x : TensorInputs ℝ)
    (tile : Tile) (r v : Nat) (hk : tile.key=0) (fits : s.K ≤ s.BK)
    (hr : r<128) (ht : tile.chunk*128+r<s.T) (hv : tile.value*s.BV+v<s.V) :
    forwardLane realOps p s x tile r v=
      (outputSpec s.T s.K (selectedScale s.K s.useScale) (headInputs x tile.bh)
        (tile.chunk*128+r) (tile.value*s.BV+v),
       normalizerSpec s.T s.K (selectedScale s.K s.useScale) (headInputs x tile.bh)
        (tile.chunk*128+r)) := by
  let w := fun j => weight s.K (selectedScale s.K s.useScale) (headInputs x tile.bh) (tile.chunk*128+r) j
  let f := fun j => w j*x.v tile.bh j (tile.value*s.BV+v)
  have prior : foldN (forwardStep realOps p s x tile r v false) (tile.chunk*4) (0,0)=
      (∑ j∈Finset.range (tile.chunk*128),f j,∑ j∈Finset.range (tile.chunk*128),w j) := by
    rw [foldN_bounded_congr _ (fun j carry =>
      (carry.1+∑ u∈Finset.range 32,f (j*32+u),carry.2+∑ u∈Finset.range 32,w (j*32+u))) _ _
      (by intro j hj carry;simpa [w,f] using forwardStep_real p s x tile r v j false carry hk fits ht hv (by intro _;exact hj))]
    rw [foldN_pair_add,sum_tiles,sum_tiles]
    simp [show tile.chunk*4*32=tile.chunk*128 by omega]
  have diagonal : ∀ carry : ℝ×ℝ,
      foldN (forwardStep realOps p s x tile r v true) 4 carry=
      (carry.1+∑ j∈Finset.range 128,f (tile.chunk*128+j),
       carry.2+∑ j∈Finset.range 128,w (tile.chunk*128+j)) := by
    intro carry
    rw [foldN_bounded_congr _ (fun j c =>
      (c.1+∑ u∈Finset.range 32,f (tile.chunk*128+(j*32+u)),
       c.2+∑ u∈Finset.range 32,w (tile.chunk*128+(j*32+u)))) _ _
      (by intro j hj c;simpa [w,f,Nat.add_assoc] using forwardStep_real p s x tile r v j true c hk fits ht hv (by simp))]
    rw [foldN_pair_add, sum_tiles 4 32 (fun j => f (tile.chunk*128+j)),
      sum_tiles 4 32 (fun j => w (tile.chunk*128+j))]
  unfold forwardLane
  simp only [real_zero]
  rw [prior,diagonal]
  simp only [real_cast]
  apply Prod.ext <;> dsimp only
  · rw [← Finset.sum_range_add]
    change (∑ j∈Finset.range (tile.chunk*128+128),weight _ _ _ _ j * _)=outputSpec _ _ _ _ _ _
    unfold outputSpec
    rw [causal_sum_prefix _ _ _ _ _ _ (by omega),causal_sum_prefix _ _ _ _ _ _ (by omega)]
    rfl
  · rw [← Finset.sum_range_add]
    change (∑ j∈Finset.range (tile.chunk*128+128),w j)=∑ j∈Finset.range s.T,w j
    have h1 := causal_sum_prefix s.K (tile.chunk*128+r) (tile.chunk*128+128)
      (selectedScale s.K s.useScale) (headInputs x tile.bh) (fun _ => 1) (by omega)
    have h2 := causal_sum_prefix s.K (tile.chunk*128+r) s.T
      (selectedScale s.K s.useScale) (headInputs x tile.bh) (fun _ => 1) (by omega)
    simpa [w] using h1.trans h2.symm

#axiomsClean forwardScore_real
#axiomsClean forwardStep_real
#axiomsClean forward_whole_real
end
end VeriTile.Bench.Repairs.ParallelAttention
