import bench.tritonbench_g.int_scaled_matmul.Proofs.ScaledWordComplete
namespace VeriTile.Bench.TritonBenchG.IntScaledMatmul.ScaledWord
open VeriTile.Triton FP.Structural FP.IntegerConversion
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
local instance scaledContractDefault : Inhabited Word := ⟨FP.Value.zero .fp32⟩
/-- Independent unbounded sum used only as a specification: the executable
backend uses finite words. The value theorem projects this sum to signed32
before any floating conversion, including when the sum overflows int32. -/
def blockSum (p : Parameters) (s : State Word) (k : Nat) (i : CIndex p) : Int :=
  (List.finRange p.blockInner).foldl
    (fun total t => total + aValue p s k (i.1,t,PUnit.unit)*bValue p s k (t,i.2.1,PUnit.unit)) 0
def exactAccumulation (p : Parameters) (s : State Word) : Nat → CIndex p → Int
  | 0 => fun _ => 0
  | k+1 => fun i => exactAccumulation p s k i + blockSum p s k i

theorem dot_low32 (p : Parameters) (s : State Word) (k : Nat) (i : CIndex p) :
    FP.IntegerDot.wordDot [] (aValue p s k) (bValue p s k) i = BitVec.ofInt 32 (blockSum p s k i) := by
  rcases i with ⟨r,c,u⟩
  cases u
  exact FP.IntegerDot.fold_is_low32 _ _ _ 0

theorem accumulation_low32 (p : Parameters) (s : State Word) (n : Nat) (i : CIndex p) :
    BitVec.ofInt 32 (accumulation p s n i) = BitVec.ofInt 32 (exactAccumulation p s n i) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [accumulation, wrap, BitVec.ofInt_toInt, exactAccumulation, BitVec.ofInt_add]
    rw [ih, dot_low32]

theorem accumulation_value (p : Parameters) (s : State Word) (n : Nat) (i : CIndex p) :
    accumulation p s n i = wrap 32 (exactAccumulation p s n i) := by
  have h := congrArg BitVec.toInt (accumulation_low32 p s n i)
  cases n with
  | zero => rfl
  | succ n =>
    simpa only [accumulation, wrap, BitVec.ofInt_toInt] using h

theorem expectedSpec (cfg : FP.Config) (p : Parameters) (initial : State Word)
    (hd : InputDefined cfg p initial) (i : CIndex p) (hi : active p initial i) :
    ∃ q, (scaled cfg p initial i).toRat?=some q ∧
      TowardZero q (expected cfg p initial i) ∧ SignedRange 32 (expected cfg p initial i) := by
  have h := hd i hi
  cases hx : toSigned 32 (scaled cfg p initial i) with
  | none => simp [hx] at h
  | some z => simpa [expected,hx] using (toSigned_spec 32 (scaled cfg p initial i) z).mp hx

/-- Coverage of the masked branch: every K position has an active block
lane. The unmasked EVEN_K branch is preserved without asserting that its
host heuristic makes all loads in bounds. -/
theorem inner_coverage (p : Parameters) (s : State Word)
    (hcover : p.inner ≤ p.steps*p.blockInner) (j : Fin p.inner) :
    ∃ k, ∃ lane : Fin p.blockInner, k<p.steps ∧
      (p.even=Bool.true ∨ inputInner p s lane < p.inner-k*p.blockInner) ∧
      k*p.blockInner+inputInner p s lane=j.val := by
  have hk : 0<p.blockInner := by
    by_contra hn
    have hz : p.blockInner=0 := by omega
    simp [hz] at hcover
    have hj := j.isLt
    omega
  let lane : Fin p.blockInner := ⟨j.val%p.blockInner, Nat.mod_lt _ hk⟩
  have hdiv : j.val/p.blockInner < p.steps := (Nat.div_lt_iff_lt_mul hk).mpr (lt_of_lt_of_le j.isLt hcover)
  have heq : j.val/p.blockInner*p.blockInner+j.val%p.blockInner=j.val := Nat.div_add_mod' _ _
  refine ⟨j.val/p.blockInner,lane,hdiv,Or.inr ?_,?_⟩ <;> simp only [inputInner,lane]
  · omega
  · exact heq

/-- The actual contiguous suffix is injective on active lanes; no extra
user-supplied address-injectivity premise is needed. -/
theorem output_injective (p : Parameters) (s : State Word) :
    ∀ i j, active p s i → active p s j → outputOffset p s i=outputOffset p s j → i=j := by
  exact smCAddr_inj_active p.rows p.columns p.blockRows p.blockColumns (rowBlock p s) (columnBlock p s)

theorem even_lane_in_bounds (p : Parameters)
    (he : p.even=Bool.true → p.inner=p.steps*p.blockInner)
    (heven : p.even=Bool.true) (k : Nat) (hk : k<p.steps) (j : Fin p.blockInner) :
    k*p.blockInner+j.val < p.inner := by
  have hmul := Nat.mul_le_mul_right p.blockInner (Nat.succ_le_of_lt hk)
  rw [Nat.succ_mul] at hmul
  have hinner := he heven
  omega

#axiomsClean dot_low32
#axiomsClean accumulation_low32
#axiomsClean accumulation_value
#axiomsClean expectedSpec
#axiomsClean inner_coverage
#axiomsClean output_injective
#axiomsClean even_lane_in_bounds
end
end VeriTile.Bench.TritonBenchG.IntScaledMatmul.ScaledWord
