import bench.repairs.tritonbench_g.triton_attention.Whole001.Proofs.MathDerivative

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Mathematical
open scoped BigOperators
noncomputable section

/-- Independent finite causal softmax. The index bound is the query row,
not a rounded-up source tile extent. -/
def score (dim : Nat) (scale : ℝ) (q k : Nat → Nat → ℝ) (i j : Nat) : ℝ :=
  (∑ d ∈ Finset.range dim, q i d*k j d)*scale
def attention (dim : Nat) (scale : ℝ) (q k v : Nat → Nat → ℝ) (i e : Nat) : ℝ :=
  (∑ j ∈ Finset.range (i+1), Real.exp (score dim scale q k i j)*v j e) /
    (∑ j ∈ Finset.range (i+1), Real.exp (score dim scale q k i j))

def mass (row stop : Nat) (score : Nat → ℝ) (m : ℝ) : ℝ :=
  ∑ j ∈ Finset.range stop, if j ≤ row then Real.exp (score j-m) else 0
def numerator (row stop : Nat) (score value : Nat → ℝ) (m : ℝ) : ℝ :=
  ∑ j ∈ Finset.range stop, if j ≤ row then Real.exp (score j-m)*value j else 0
def blockMass (row start width : Nat) (score : Nat → ℝ) (m : ℝ) : ℝ :=
  ∑ j : Fin width, if start+j.val ≤ row then Real.exp (score (start+j.val)-m) else 0
def blockNumerator (row start width : Nat) (score value : Nat → ℝ) (m : ℝ) : ℝ :=
  ∑ j : Fin width, if start+j.val ≤ row then Real.exp (score (start+j.val)-m)*value (start+j.val) else 0

theorem mass_rescale (row stop : Nat) (score : Nat → ℝ) (a b : ℝ) :
    mass row stop score a * Real.exp (a-b) = mass row stop score b := by
  simp only [mass,Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro j hj
  split_ifs
  · rw [←Real.exp_add]
    congr 1
    ring
  · simp

theorem numerator_rescale (row stop : Nat) (score value : Nat → ℝ) (a b : ℝ) :
    numerator row stop score value a * Real.exp (a-b) = numerator row stop score value b := by
  simp only [numerator,Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro j hj
  split_ifs
  · rw [mul_right_comm,←Real.exp_add]
    congr 2
    ring
  · simp

theorem mass_step (row start width : Nat) (score : Nat → ℝ) (m : ℝ) :
    mass row (start+width) score m = mass row start score m + blockMass row start width score m := by
  simp only [mass,blockMass,Finset.sum_range_add]
  rw [Fin.sum_univ_eq_sum_range (fun j : Nat => if start+j ≤ row then Real.exp (score (start+j)-m) else 0) width]

theorem numerator_step (row start width : Nat) (score value : Nat → ℝ) (m : ℝ) :
    numerator row (start+width) score value m =
      numerator row start score value m + blockNumerator row start width score value m := by
  simp only [numerator,blockNumerator,Finset.sum_range_add]
  rw [Fin.sum_univ_eq_sum_range (fun j : Nat => if start+j ≤ row then Real.exp (score (start+j)-m)*value (start+j) else 0) width]

theorem mass_pos (row stop : Nat) (score : Nat → ℝ) (m : ℝ) (hstop : 0 < stop) :
    0 < mass row stop score m := by
  apply Finset.sum_pos'
  · intro j hj
    split_ifs <;> positivity
  · refine ⟨0,Finset.mem_range.mpr hstop,?_⟩
    simp [Real.exp_pos]

/-- The normalized source update works for every finite row maximum. This
does not assume a source reduction result or require native arithmetic laws. -/
theorem normalized_step (row start width : Nat) (score value : Nat → ℝ)
    (oldMax newMax : ℝ) (hstart : 0 < start) :
    let oldL := mass row start score oldMax * Real.exp (oldMax-newMax)
    let newL := blockMass row start width score newMax + oldL
    numerator row start score value oldMax / mass row start score oldMax * (oldL/newL) +
      blockNumerator row start width score value newMax / newL =
      numerator row (start+width) score value newMax / mass row (start+width) score newMax := by
  dsimp only
  rw [mass_rescale]
  rw [mass_step,numerator_step]
  have hn := ne_of_gt (mass_pos row start score oldMax hstart)
  rw [←mass_rescale row start score oldMax newMax]
  have he := numerator_rescale row start score value oldMax newMax
  field_simp
  simp only [he,add_comm]

theorem mass_done (row stop : Nat) (score : Nat → ℝ) (m : ℝ) (h : row+1 ≤ stop) :
    mass row stop score m = ∑ j ∈ Finset.range (row+1), Real.exp (score j-m) := by
  rw [show stop=(row+1)+(stop-(row+1)) by omega,mass,Finset.sum_range_add]
  have first : (∑ j ∈ Finset.range (row+1), if j ≤ row then Real.exp (score j-m) else 0) =
      ∑ j ∈ Finset.range (row+1), Real.exp (score j-m) := by
    apply Finset.sum_congr rfl
    intro j hj
    rw [if_pos (by have := Finset.mem_range.mp hj; omega)]
  simp [first,show ∀ j, ¬ row+1+j ≤ row by omega]

theorem numerator_done (row stop : Nat) (score value : Nat → ℝ) (m : ℝ) (h : row+1 ≤ stop) :
    numerator row stop score value m =
      ∑ j ∈ Finset.range (row+1), Real.exp (score j-m)*value j := by
  rw [show stop=(row+1)+(stop-(row+1)) by omega,numerator,Finset.sum_range_add]
  have first : (∑ j ∈ Finset.range (row+1), if j ≤ row then Real.exp (score j-m)*value j else 0) =
      ∑ j ∈ Finset.range (row+1), Real.exp (score j-m)*value j := by
    apply Finset.sum_congr rfl
    intro j hj
    rw [if_pos (by have := Finset.mem_range.mp hj; omega)]
  simp [first,show ∀ j, ¬ row+1+j ≤ row by omega]

theorem normalized_done (dim : Nat) (scale : ℝ) (q k v : Nat → Nat → ℝ)
    (row col stop : Nat) (m : ℝ) (h : row+1 ≤ stop) :
    numerator row stop (score dim scale q k row) (fun j => v j col) m /
      mass row stop (score dim scale q k row) m = attention dim scale q k v row col := by
  rw [mass_done _ _ _ _ h,numerator_done _ _ _ _ _ h]
  simp only [Real.exp_sub,div_mul_eq_mul_div,←Finset.sum_div,attention]
  field_simp

#axiomsClean mass_rescale
#axiomsClean numerator_rescale
#axiomsClean normalized_step
#axiomsClean normalized_done
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Mathematical
