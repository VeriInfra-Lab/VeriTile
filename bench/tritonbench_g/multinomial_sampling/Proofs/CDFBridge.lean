import bench.tritonbench_g.multinomial_sampling.Proofs.Real

namespace VeriTile.Bench.Repairs.MultinomialSampling.CDFBridge
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MultinomialSampling.SourceValue
open scoped BigOperators
noncomputable section
set_option maxRecDepth 20000
set_option maxHeartbeats 3000000

abbrev P := Real.primitives

def weights (p : Parameters) (s : State ℝ) (i : Row p) (j : Nat) : ℝ :=
  if j < p.tokens then (s.mem p.scores (row p s i * p.strideScoreBatch + j * p.strideScoreToken)).read .real else 0

def labels (p : Parameters) (s : State ℝ) (i : Row p) (j : Nat) : Int :=
  (s.mem p.indices (row p s i * p.strideIndexBatch + j * p.strideIndexToken)).read .int

def Nonnegative (p : Parameters) (s : State ℝ) (i : Row p) : Prop :=
  ∀ j, j < p.tokens → 0 ≤ weights p s i j

theorem weights_nonnegative (p : Parameters) (s : State ℝ) (i : Row p)
    (h : Nonnegative p s i) (j : Nat) : 0 ≤ weights p s i j := by
  by_cases hj : j < p.tokens
  · exact h j hj
  · simp [weights,hj]

theorem scores_eq (p : Parameters) (s : State ℝ) (i : Row p) (ha : active p s i)
    (b : Nat) (j : Fin p.blockTokens) :
    scores P .fp32 p s b (i.1,j,PUnit.unit) = weights p s i (b+j.val) := by
  have hi : (i.1, PUnit.unit) = i := by cases i; rfl
  simp [scores, P, Real.primitives, weights, scoreOffset, hi, ha, readScore, padding]

theorem scanned_eq (p : Parameters) (s : State ℝ) (i : Row p) (ha : active p s i)
    (b : Nat) (j : Fin p.blockTokens) :
    scanned P .fp32 p s b (i.1,j,PUnit.unit) =
      Spec.cumulative (fun k => weights p s i (b+k)) (j.val+1) := by
  change (∑ k ∈ Finset.range (j.val+1),
    if h : k < p.blockTokens then scores P .fp32 p s b (i.1,⟨k,h⟩,PUnit.unit) else 0) = _
  unfold Spec.cumulative
  apply Finset.sum_congr rfl
  intro k hk
  have h : k < p.blockTokens := by have := j.isLt; have := Finset.mem_range.mp hk; omega
  simp only [h, ↓reduceDIte, scores_eq p s i ha]

theorem maximum_scanned (p : Parameters) (s : State ℝ) (i : Row p) (ha : active p s i)
    (hn : Nonnegative p s i) (b : Nat) :
    P.maximum (some .fp32) ⟨1,by simp⟩ Bool.false (scanned P .fp32 p s b) i =
      Spec.cumulative (fun k => weights p s i (b+k)) p.blockTokens := by
  let last : Fin p.blockTokens := ⟨p.blockTokens-1, by have := p.tokensDomain.1; omega⟩
  change Real.maximum (fun k : Fin p.blockTokens => scanned P .fp32 p s b (i.1,k,PUnit.unit)) = _
  rw [Real.maximum_eq p.tokensDomain.1 _ last]
  · rw [scanned_eq p s i ha]
    congr 1
    dsimp [last]
    have := p.tokensDomain.1
    omega
  · intro k
    rw [scanned_eq p s i ha, scanned_eq p s i ha]
    apply Real.cumulative_mono
    · exact fun j => weights_nonnegative p s i hn (b+j)
    · have := k.isLt; dsimp [last]; omega

theorem rounds_acc (p : Parameters) (s : State ℝ) (i : Row p) (ha : active p s i)
    (hn : Nonnegative p s i) (k : Nat) :
    (rounds P .fp32 p s k).acc i = Spec.cumulative (weights p s i) (k*p.blockTokens) := by
  induction k with
  | zero => simp [rounds, initialAccumulator, P, Real.primitives, Spec.prefix_zero]
  | succ k ih =>
    change (rounds P .fp32 p s k).acc i +
      P.maximum (some .fp32) ⟨1,by simp⟩ Bool.false (scanned P .fp32 p s (k*p.blockTokens)) i = _
    rw [ih, maximum_scanned p s i ha hn, Real.cumulative_shift]
    simp [Nat.add_mul]

theorem valid_iff (p : Parameters) (s : State ℝ) (i : Row p) (ha : active p s i)
    (hn : Nonnegative p s i) (k : Nat) (j : Fin p.blockTokens) :
    valid P .fp32 p s (k*p.blockTokens) (rounds P .fp32 p s k) (i.1,j,PUnit.unit) = Bool.true ↔
      Spec.Interval (weights p s i) (sample P p s i) (k*p.blockTokens+j.val) := by
  have hi : (i.1,PUnit.unit) = i := by cases i; rfl
  have hc : cumulative P .fp32 p s (k*p.blockTokens) (rounds P .fp32 p s k) (i.1,j,PUnit.unit) =
      Spec.cumulative (weights p s i) (k*p.blockTokens+j.val+1) := by
    change (rounds P .fp32 p s k).acc (i.1,PUnit.unit) + scanned P .fp32 p s (k*p.blockTokens) (i.1,j,PUnit.unit) = _
    rw [hi, rounds_acc p s i ha hn, scanned_eq p s i ha, Real.cumulative_shift]
    congr 1
  change (decide (cumulative P .fp32 p s (k*p.blockTokens) (rounds P .fp32 p s k) (i.1,j,PUnit.unit) -
      scores P .fp32 p s (k*p.blockTokens) (i.1,j,PUnit.unit) < sample P p s (i.1,PUnit.unit)) &&
      decide (sample P p s (i.1,PUnit.unit) ≤ cumulative P .fp32 p s (k*p.blockTokens) (rounds P .fp32 p s k) (i.1,j,PUnit.unit))) = Bool.true ↔ _
  rw [hc, scores_eq p s i ha, hi]
  simp only [Bool.and_eq_true, decide_eq_true_eq]
  unfold Spec.Interval
  rw [Spec.prefix_succ, add_sub_cancel_right]

theorem found_iff (p : Parameters) (s : State ℝ) (i : Row p) (ha : active p s i)
    (hn : Nonnegative p s i) (k : Nat) :
    found P .fp32 p s (k*p.blockTokens) (rounds P .fp32 p s k) i = Bool.true ↔
      ∃ j : Fin p.blockTokens, Spec.Interval (weights p s i) (sample P p s i) (k*p.blockTokens+j.val) := by
  change decide (0 < TileShape.boolCount32 (fun j : Fin p.blockTokens =>
    valid P .fp32 p s (k*p.blockTokens) (rounds P .fp32 p s k) (i.1,j,PUnit.unit))) = Bool.true ↔ _
  rw [decide_eq_true_eq, Real.boolCount_positive]
  · exact exists_congr (fun j => valid_iff p s i ha hn k j)
  · have := p.tokensDomain.2.1
    omega

theorem interval_inside (p : Parameters) (s : State ℝ) (i : Row p) (u : ℝ) (j : Nat)
    (hj : Spec.Interval (weights p s i) u j) : j < p.tokens := by
  by_contra h
  have hz : weights p s i j = 0 := by simp [weights,h]
  have he := Spec.prefix_succ (weights p s i) j
  rw [hz, add_zero] at he
  unfold Spec.Interval at hj
  rw [he] at hj
  exact (not_lt_of_ge hj.2 hj.1)

theorem interval_unique (p : Parameters) (s : State ℝ) (i : Row p)
    (hn : Nonnegative p s i) (u : ℝ) (j t : Nat)
    (hj : Spec.Interval (weights p s i) u j) (ht : Spec.Interval (weights p s i) u t) : j = t :=
  Spec.interval_unique p.tokens (weights p s i) hn u j t
    (interval_inside p s i u j hj) (interval_inside p s i u t ht) hj ht

theorem found_chosen (p : Parameters) (s : State ℝ) (i : Row p) (ha : active p s i)
    (hn : Nonnegative p s i) (j : Nat)
    (hj : Spec.Interval (weights p s i) (sample P p s i) j) (k : Nat) :
    found P .fp32 p s (k*p.blockTokens) (rounds P .fp32 p s k) i = Bool.true ↔
      k*p.blockTokens ≤ j ∧ j < (k+1)*p.blockTokens := by
  rw [found_iff p s i ha hn]
  constructor
  · rintro ⟨t,ht⟩
    have eq := interval_unique p s i hn _ _ _ ht hj
    have := t.isLt
    rw [Nat.add_mul]
    simp only [Nat.one_mul]
    omega
  · rintro ⟨hlo,hhi⟩
    have ht : j-k*p.blockTokens < p.blockTokens := by
      rw [Nat.add_mul] at hhi
      simp only [Nat.one_mul] at hhi
      omega
    refine ⟨⟨j-k*p.blockTokens,ht⟩, ?_⟩
    have eq : k*p.blockTokens+(j-k*p.blockTokens) = j := by omega
    simpa only [eq] using hj

theorem selected_chosen (p : Parameters) (s : State ℝ) (i : Row p) (ha : active p s i)
    (hn : Nonnegative p s i) (j : Nat)
    (hj : Spec.Interval (weights p s i) (sample P p s i) j) (k : Nat)
    (hf : found P .fp32 p s (k*p.blockTokens) (rounds P .fp32 p s k) i = Bool.true) :
    selected P .fp32 p s (k*p.blockTokens) (rounds P .fp32 p s k) i = j := by
  have hex : ∃ t : Fin p.blockTokens,
      valid P .fp32 p s (k*p.blockTokens) (rounds P .fp32 p s k) (i.1,t,PUnit.unit) = Bool.true := by
    obtain ⟨t,ht⟩ := (found_iff p s i ha hn k).mp hf
    exact ⟨t,(valid_iff p s i ha hn k t).mpr ht⟩
  obtain ⟨t,ht,hv⟩ := Real.argmax_true p.tokensDomain.1 _ hex
  change k*p.blockTokens + TileShape.leftmostIntArgMax (fun t : Fin p.blockTokens =>
    if valid P .fp32 p s (k*p.blockTokens) (rounds P .fp32 p s k) (i.1,t,PUnit.unit) then (1:Int) else 0) = j
  rw [ht]
  exact interval_unique p s i hn _ _ _ ((valid_iff p s i ha hn k t).mp hv) hj

theorem initial_output (p : Parameters) (s : State ℝ) (i : Row p) (ha : active p s i) :
    (rounds P .fp32 p s 0).output i = labels p s i 0 := by
  simp [rounds,initialAccumulator,initialOutput,intLoad,ha,labels]

theorem rounds_output_chosen (p : Parameters) (s : State ℝ) (i : Row p) (ha : active p s i)
    (hn : Nonnegative p s i) (j : Nat)
    (hj : Spec.Interval (weights p s i) (sample P p s i) j) (k : Nat) :
    (rounds P .fp32 p s k).output i = if j < k*p.blockTokens then labels p s i j else labels p s i 0 := by
  induction k with
  | zero => simpa using initial_output p s i ha
  | succ k ih =>
    change (if found P .fp32 p s (k*p.blockTokens) (rounds P .fp32 p s k) i then
      selectedIndex P .fp32 p s (k*p.blockTokens) (rounds P .fp32 p s k) i else
      (rounds P .fp32 p s k).output i) = _
    by_cases hf : found P .fp32 p s (k*p.blockTokens) (rounds P .fp32 p s k) i = Bool.true
    · have hb := (found_chosen p s i ha hn j hj k).mp hf
      have hs := selected_chosen p s i ha hn j hj k hf
      have inside := interval_inside p s i _ j hj
      have clamp : min j (p.tokens-1) = j := Nat.min_eq_left (by omega)
      simp [hf, selectedIndex, ha, hs, hb.2, labels,clamp]
    · have hb := (found_chosen p s i ha hn j hj k).not.mp hf
      have eq : (j < k*p.blockTokens) ↔ j < (k+1)*p.blockTokens := by
        rw [Nat.add_mul] at *
        simp only [Nat.one_mul] at *
        omega
      simp [hf, ih, eq]

theorem rounds_output_zero (p : Parameters) (s : State ℝ) (i : Row p) (ha : active p s i)
    (hn : Nonnegative p s i) (hu : sample P p s i = 0) (k : Nat) :
    (rounds P .fp32 p s k).output i = labels p s i 0 := by
  induction k with
  | zero => exact initial_output p s i ha
  | succ k ih =>
    have hf : found P .fp32 p s (k*p.blockTokens) (rounds P .fp32 p s k) i ≠ Bool.true := by
      intro hf
      obtain ⟨t,ht⟩ := (found_iff p s i ha hn k).mp hf
      have hp := Real.cumulative_nonnegative (weights p s i) (weights_nonnegative p s i hn) (k*p.blockTokens+t.val)
      rw [hu] at ht
      exact not_lt_of_ge hp ht.1
    change (if found P .fp32 p s (k*p.blockTokens) (rounds P .fp32 p s k) i then _ else _) = _
    simpa only [hf, ↓reduceIte] using ih

theorem count_covers (p : Parameters) : p.tokens ≤ count p * p.blockTokens := by
  have h := p.tokensDomain.1
  have hd := Nat.mod_lt (p.tokens + p.blockTokens - 1) h
  have he := Nat.mod_add_div (p.tokens + p.blockTokens - 1) p.blockTokens
  rw [Nat.mul_comm p.blockTokens] at he
  unfold count
  omega

/-- Complete source recurrence meets the input-only inverse-CDF relation.
No scan, argmax, output, or probability oracle is an assumption. -/
theorem rounds_result (p : Parameters) (s : State ℝ) (i : Row p) (ha : active p s i)
    (domain : Spec.Domain p.tokens (weights p s i)) :
    ∃ j, Spec.Position p.tokens (weights p s i) (sample P p s i) j ∧
      (rounds P .fp32 p s (count p)).output i = labels p s i j := by
  have hn : Nonnegative p s i := domain.2.1
  have hu : 0 ≤ sample P p s i ∧ sample P p s i < 1 := by
    rw [Real.primitive_sample]
    have h := Random.randReal_range (seedValue p s i) (offsetValue p s i) .bits32 10
    exact ⟨h.1,h.2.2⟩
  obtain ⟨j,hj⟩ := Spec.position_exists p.tokens (weights p s i) domain _ hu
  refine ⟨j,hj,?_⟩
  have hjt := hj.1
  rcases hj.2 with hz | hp
  · rw [hz.2]
    exact rounds_output_zero p s i ha hn hz.1 (count p)
  · rw [rounds_output_chosen p s i ha hn j hp.2 (count p)]
    have hc := count_covers p
    exact if_pos (by omega)

#axiomsClean rounds_acc
#axiomsClean valid_iff
#axiomsClean found_iff
#axiomsClean selected_chosen
#axiomsClean rounds_result
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MultinomialSampling.CDFBridge
