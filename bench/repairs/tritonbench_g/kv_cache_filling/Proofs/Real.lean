import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.Store

namespace VeriTile.Bench.Repairs.KvCacheFilling

/-- Independent full-head extrema, stated only in terms of input coordinates. -/
structure Extrema (n : Nat) (x : Nat → ℝ) (lo hi : ℝ) : Prop where
  lower : ∀ i, i < n → lo ≤ x i
  upper : ∀ i, i < n → x i ≤ hi
  attains_low : ∃ i, i < n ∧ x i = lo
  attains_high : ∃ i, i < n ∧ x i = hi

theorem Tree.eval_min_le (t : Tree) (x : Nat → ℝ) (i : Nat) (h : i ∈ t.leaves) :
    t.eval min x ≤ x i := by
  induction t with
  | leaf j =>
    simp only [Tree.leaves, List.mem_singleton] at h
    subst i
    exact le_refl _
  | node a b ia ib =>
    rcases List.mem_append.mp h with ha | hb
    · exact le_trans (min_le_left _ _) (ia ha)
    · exact le_trans (min_le_right _ _) (ib hb)

theorem Tree.le_eval_min (t : Tree) (x : Nat → ℝ) (lo : ℝ)
    (h : ∀ i ∈ t.leaves, lo ≤ x i) : lo ≤ t.eval min x := by
  induction t with
  | leaf i => exact h i (by simp [Tree.leaves])
  | node a b ia ib =>
    exact le_min (ia (by intro i hi; exact h i (List.mem_append_left _ hi)))
      (ib (by intro i hi; exact h i (List.mem_append_right _ hi)))

theorem Tree.le_eval_max (t : Tree) (x : Nat → ℝ) (i : Nat) (h : i ∈ t.leaves) :
    x i ≤ t.eval max x := by
  induction t with
  | leaf j =>
    simp only [Tree.leaves, List.mem_singleton] at h
    subst i
    exact le_refl _
  | node a b ia ib =>
    rcases List.mem_append.mp h with ha | hb
    · exact le_trans (ia ha) (le_max_left _ _)
    · exact le_trans (ib hb) (le_max_right _ _)

theorem Tree.eval_max_le (t : Tree) (x : Nat → ℝ) (hi : ℝ)
    (h : ∀ i ∈ t.leaves, x i ≤ hi) : t.eval max x ≤ hi := by
  induction t with
  | leaf i => exact h i (by simp [Tree.leaves])
  | node a b ia ib =>
    exact max_le (ia (by intro i hi; exact h i (List.mem_append_left _ hi)))
      (ib (by intro i hi; exact h i (List.mem_append_right _ hi)))

theorem Plan.extrema {n : Nat} (p : Plan n) (x : Nat → ℝ) (lo hi : ℝ) (h : Extrema n x lo hi) :
    p.low.eval min x = lo ∧ p.high.eval max x = hi := by
  constructor
  · apply le_antisymm
    · obtain ⟨i,hi,e⟩ := h.attains_low
      simpa [e] using p.low.eval_min_le x i (p.low_covers i hi)
    · exact p.low.le_eval_min x lo (by intro i hi; exact h.lower i (p.low_valid i hi))
  · apply le_antisymm
    · exact p.high.eval_max_le x hi (by intro i hi; exact h.upper i (p.high_valid i hi))
    · obtain ⟨i,hi,e⟩ := h.attains_high
      simpa [e] using p.high.le_eval_max x i (p.high_covers i hi)

/-- Ideal uniform affine quantization, including the repaired constant-row case.
The formula is independent of the execution tree and primitive expression graph. -/
noncomputable def idealScale (levels : Nat) (lo hi : ℝ) : ℝ := if hi = lo then 1 else (hi - lo) / levels
noncomputable def idealCode (levels : Nat) (lo hi x : ℝ) : BitVec 8 :=
  BitVec.ofNat 8 (Int.floor ((x - lo) / idealScale levels lo hi + 1 / 2)).toNat

def packByte (low high : BitVec 8) : BitVec 8 := low + high * 16

noncomputable def realOps : Operations ℝ (BitVec 8) where
  minimum := min
  maximum := max
  sub := (· - ·)
  div := (· / ·)
  add := (· + ·)
  neg := Neg.neg
  equal := fun a b => decide (a = b)
  literal := fun n => n
  half := 1 / 2
  uint8 := fun x => BitVec.ofNat 8 (Int.floor x).toNat
  pack := packByte

theorem idealScale_pos {n levels : Nat} (hl : 0 < levels) {x : Nat → ℝ} {lo hi : ℝ}
    (h : Extrema n x lo hi) : 0 < idealScale levels lo hi := by
  obtain ⟨i,hin,he⟩ := h.attains_low
  have hle : lo ≤ hi := he ▸ h.upper i hin
  unfold idealScale
  split_ifs with heq
  · norm_num
  · exact div_pos (sub_pos.mpr (lt_of_le_of_ne hle (Ne.symm heq))) (by exact_mod_cast hl)

theorem ideal_argument_nonnegative {n levels : Nat} (hl : 0 < levels) {x : Nat → ℝ} {lo hi : ℝ}
    (h : Extrema n x lo hi) (i : Nat) (hi' : i < n) :
    0 ≤ (x i - lo) / idealScale levels lo hi + 1 / 2 := by
  have hs := idealScale_pos hl h
  have hd := div_nonneg (sub_nonneg.mpr (h.lower i hi')) (le_of_lt hs)
  linarith

theorem quantize_real {n : Nat} (p : Plan n) (levels : Nat) (x : Nat → ℝ) (lo hi : ℝ)
    (h : Extrema n x lo hi) :
    quantize realOps levels x p.low p.high =
      ⟨fun i => idealCode levels lo hi (x i), idealScale levels lo hi, -lo / idealScale levels lo hi⟩ := by
  obtain ⟨hl,hh⟩ := p.extrema x lo hi h
  simp [quantize, realOps, hl, hh, idealScale, idealCode, add_div, neg_div, sub_eq_add_neg]

noncomputable def idealRow (policy : Policy) (width : Nat) (x : Nat → ℝ) (lo hi : ℝ) : RowResult ℝ (BitVec 8) :=
  match policy with
  | .copy => ⟨fun i => .fp32 (x i), 0, 0⟩
  | .int8 => ⟨fun i => .uint8 (idealCode 255 lo hi (x i)), idealScale 255 lo hi, -lo / idealScale 255 lo hi⟩
  | .int4 => ⟨fun i => .uint8 (packByte (idealCode 15 lo hi (x i)) (idealCode 15 lo hi (x (i + width)))),
      idealScale 15 lo hi, -lo / idealScale 15 lo hi⟩

theorem rowKernel_real {n : Nat} (p : Plan n) (policy : Policy) (width : Nat) (x : Nat → ℝ)
    (lo hi : ℝ) (h : Extrema n x lo hi) :
    rowKernel realOps policy width x p.low p.high = idealRow policy width x lo hi := by
  cases policy with
  | copy => simp [rowKernel, idealRow, realOps]
  | int4 => simp only [rowKernel, idealRow, quantize_real p 15 x lo hi h] <;> rfl
  | int8 => simp only [rowKernel, idealRow, quantize_real p 255 x lo hi h] <;> rfl

/-- One statement covers every active token, head and K/V/cache-metadata role. -/
theorem whole_real_readback (r : Request) (d : Domain r) (kp : Plan r.dk) (vp : Plan r.dv)
    (k v : Nat → Nat → ℝ) (m : Address → Cell ℝ (BitVec 8)) (j : Job) (hj : j ∈ jobs r)
    (a : Address) (ha : a ∈ rowAddresses r j) (kl kh vl vh : ℝ)
    (hk : Extrema r.dk (k (sourceRow r j)) kl kh) (hv : Extrema r.dv (v (sourceRow r j)) vl vh) :
    execute r realOps kp vp k v m a =
      rowValue (idealRow r.policy r.ck (k (sourceRow r j)) kl kh)
        (idealRow r.policy r.cv (v (sourceRow r j)) vl vh) a := by
  rw [whole_readback r d realOps kp vp k v m j hj a ha]
  simp only [taskValue, rowKernel_real kp _ _ _ kl kh hk, rowKernel_real vp _ _ _ vl vh hv]

end VeriTile.Bench.Repairs.KvCacheFilling
