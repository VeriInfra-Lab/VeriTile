import bench.repairs.tritonbench_g.triton_attention.Whole001.Proofs.Math

/-!
The normalized online-softmax update used by `_fwd_kernel`, over arbitrary
nonempty disjoint blocks of visible keys.  Each key block of a causal query
processed by the source is nonempty: earlier blocks are entirely visible and
the last block contains at least its first key.  The arithmetic theorem is
independent of how source indices establish that partition.
-/

namespace VeriTile.Bench.Repairs.TritonAttention.Whole001.Math

open scoped BigOperators

variable {N D : Nat}

noncomputable def blockMass (s : Fin N → ℝ) (A : Finset (Fin N)) (m : ℝ) : ℝ :=
  ∑ j ∈ A, Real.exp (s j - m)

noncomputable def blockNumerator (s : Fin N → ℝ) (V : Tensor N D)
    (A : Finset (Fin N)) (m : ℝ) (e : Fin D) : ℝ :=
  ∑ j ∈ A, Real.exp (s j - m) * V j e

theorem blockMass_pos (s : Fin N → ℝ) (A : Finset (Fin N)) (hA : A.Nonempty)
    (m : ℝ) : 0 < blockMass s A m := by
  apply Finset.sum_pos'
  · intro j _
    exact (Real.exp_pos _).le
  · obtain ⟨j, hj⟩ := hA
    exact ⟨j, hj, Real.exp_pos _⟩

theorem blockMass_rescale (s : Fin N → ℝ) (A : Finset (Fin N)) (a b : ℝ) :
    blockMass s A a * Real.exp (a - b) = blockMass s A b := by
  simp only [blockMass, Finset.sum_mul, ← Real.exp_add]
  congr 1
  funext j
  congr 1
  ring

theorem blockNumerator_rescale (s : Fin N → ℝ) (V : Tensor N D)
    (A : Finset (Fin N)) (a b : ℝ) (e : Fin D) :
    blockNumerator s V A a e * Real.exp (a - b) = blockNumerator s V A b e := by
  simp only [blockNumerator, Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro j _
  calc
    Real.exp (s j - a) * V j e * Real.exp (a - b) =
        (Real.exp (s j - a) * Real.exp (a - b)) * V j e := by ring
    _ = Real.exp (s j - b) * V j e := by rw [← Real.exp_add]; congr 2; ring

/-- The three source registers after a nonempty prefix of key blocks. -/
structure OnlineRow (D : Nat) where
  maximum : ℝ
  normalizer : ℝ
  output : Fin D → ℝ

/-- Independent closed form for a finite visible-key set. -/
@[kernel_spec] noncomputable def batchRow (s : Fin N → ℝ) (V : Tensor N D)
    (A : Finset (Fin N)) (hA : A.Nonempty) : OnlineRow D :=
  let m := A.sup' hA s
  ⟨m, blockMass s A m, fun e => blockNumerator s V A m e / blockMass s A m⟩

/-- The source update: max, rescale old L, normalize new P, rescale normalized
old accumulator, and add the normalized block contribution. -/
noncomputable def onlineStep (s : Fin N → ℝ) (V : Tensor N D) (old : OnlineRow D)
    (B : Finset (Fin N)) (hB : B.Nonempty) : OnlineRow D :=
  let m := max (B.sup' hB s) old.maximum
  let oldL := old.normalizer * Real.exp (old.maximum - m)
  let l := blockMass s B m + oldL
  ⟨m, l, fun e => old.output e * (oldL / l) +
    ∑ j ∈ B, (Real.exp (s j - m) / l) * V j e⟩

/-- The first source iteration starts with L = 0 and acc = 0; its old
`-inf` maximum contributes rescaling factor zero. -/
noncomputable def onlineFirst (s : Fin N → ℝ) (V : Tensor N D)
    (A : Finset (Fin N)) (hA : A.Nonempty) : OnlineRow D :=
  let m := A.sup' hA s
  let l := blockMass s A m + 0
  ⟨m, l, fun e => 0 * (0 / l) + ∑ j ∈ A, (Real.exp (s j - m) / l) * V j e⟩

theorem onlineFirst_eq_batchRow (s : Fin N → ℝ) (V : Tensor N D)
    (A : Finset (Fin N)) (hA : A.Nonempty) :
    onlineFirst s V A hA = batchRow s V A hA := by
  simp only [onlineFirst, batchRow, add_zero, zero_mul, zero_add]
  congr 1
  funext e
  simp only [blockNumerator, Finset.sum_div, mul_div_right_comm]

/-- Load-bearing streaming invariant: one source update preserves the exact
finite-set maximum, shifted normalizer, and normalized attention output. -/
theorem onlineStep_eq_batchRow (s : Fin N → ℝ) (V : Tensor N D)
    (A B : Finset (Fin N)) (hA : A.Nonempty) (hB : B.Nonempty)
    (hAB : Disjoint A B) :
    onlineStep s V (batchRow s V A hA) B hB =
      batchRow s V (A ∪ B) (hA.mono Finset.subset_union_left) := by
  have hm : max (B.sup' hB s) (A.sup' hA s) =
      (A ∪ B).sup' (hA.mono Finset.subset_union_left) s := by
    rw [Finset.sup'_union hA hB s]
    exact max_comm _ _
  let a := A.sup' hA s
  let m := (A ∪ B).sup' (hA.mono Finset.subset_union_left) s
  have hmass : blockMass s B m + blockMass s A a * Real.exp (a - m) =
      blockMass s (A ∪ B) m := by
    rw [blockMass_rescale]
    simp only [blockMass, Finset.sum_union hAB]
    ring
  have hnum (e : Fin D) : blockNumerator s V A m e + blockNumerator s V B m e =
      blockNumerator s V (A ∪ B) m e := by
    exact (Finset.sum_union hAB).symm
  have hne : blockMass s A a ≠ 0 := ne_of_gt (blockMass_pos s A hA a)
  simp only [onlineStep, batchRow, hm]
  change OnlineRow.mk m (blockMass s B m + blockMass s A a * Real.exp (a - m)) _ =
    OnlineRow.mk m (blockMass s (A ∪ B) m) _
  rw [hmass]
  congr 1
  funext e
  have hterm : (∑ j ∈ B, (Real.exp (s j - m) / blockMass s (A ∪ B) m) * V j e) =
      blockNumerator s V B m e / blockMass s (A ∪ B) m := by
    simp only [blockNumerator, Finset.sum_div, mul_div_right_comm]
  rw [hterm]
  calc
    blockNumerator s V A a e / blockMass s A a *
        (blockMass s A a * Real.exp (a - m) / blockMass s (A ∪ B) m) +
        blockNumerator s V B m e / blockMass s (A ∪ B) m =
        (blockNumerator s V A a e * Real.exp (a - m) + blockNumerator s V B m e) /
          blockMass s (A ∪ B) m := by field_simp
    _ = blockNumerator s V (A ∪ B) m e / blockMass s (A ∪ B) m := by
      rw [blockNumerator_rescale, hnum]

structure KeyBlock (N : Nat) where
  keys : Finset (Fin N)
  nonempty : keys.Nonempty

def collected (A : Finset (Fin N)) : List (KeyBlock N) → Finset (Fin N)
  | [] => A
  | B :: bs => collected (A ∪ B.keys) bs

/-- Pure geometric partition condition; it contains no arithmetic results. -/
def Separated (A : Finset (Fin N)) : List (KeyBlock N) → Prop
  | [] => True
  | B :: bs => Disjoint A B.keys ∧ Separated (A ∪ B.keys) bs

theorem collected_nonempty (A : Finset (Fin N)) (hA : A.Nonempty)
    (bs : List (KeyBlock N)) : (collected A bs).Nonempty := by
  induction bs generalizing A with
  | nil => exact hA
  | cons B bs ih => exact ih _ (hA.mono Finset.subset_union_left)

noncomputable def onlineFold (s : Fin N → ℝ) (V : Tensor N D)
    (old : OnlineRow D) : List (KeyBlock N) → OnlineRow D
  | [] => old
  | B :: bs => onlineFold s V (onlineStep s V old B.keys B.nonempty) bs

/-- Arbitrarily many visible key blocks preserve the closed-form invariant. -/
theorem onlineFold_eq_batchRow (s : Fin N → ℝ) (V : Tensor N D)
    (A : Finset (Fin N)) (hA : A.Nonempty) (bs : List (KeyBlock N))
    (hsep : Separated A bs) :
    onlineFold s V (batchRow s V A hA) bs =
      batchRow s V (collected A bs) (collected_nonempty A hA bs) := by
  induction bs generalizing A with
  | nil => rfl
  | cons B bs ih =>
    rw [onlineFold, onlineStep_eq_batchRow s V A B.keys hA B.nonempty hsep.1]
    exact ih _ _ hsep.2

theorem causal_batchRow (Q K V : Tensor N D) (scale : ℝ) (i : Fin N) :
    batchRow (score Q K scale i) V (causalSet i) (causalSet_nonempty i) =
      ⟨savedMax Q K scale i, savedNormalizer Q K scale i, savedOutput Q K V scale i⟩ := by
  have hl : blockMass (score Q K scale i) (causalSet i) (savedMax Q K scale i) =
      savedNormalizer Q K scale i := by
    simp only [blockMass, causalSet, Finset.sum_filter, savedNormalizer, sourceP]
  have hn (e : Fin D) :
      blockNumerator (score Q K scale i) V (causalSet i) (savedMax Q K scale i) e =
        ∑ j, sourceP Q K scale (savedMax Q K scale) i j * V j e := by
    simp only [blockNumerator, causalSet, Finset.sum_filter, sourceP]
    apply Finset.sum_congr rfl
    intro j _
    split_ifs <;> simp
  change OnlineRow.mk (savedMax Q K scale i)
    (blockMass (score Q K scale i) (causalSet i) (savedMax Q K scale i))
    (fun e => blockNumerator (score Q K scale i) V (causalSet i)
      (savedMax Q K scale i) e /
      blockMass (score Q K scale i) (causalSet i) (savedMax Q K scale i)) = _
  rw [hl]
  congr 1
  funext e
  rw [hn]
  rfl

/-- Complete independent forward theorem: the normalized streaming fold
produces the derived Y/L/M for any disjoint partition of the causal keys. -/
theorem online_forward_eq_saved (Q K V : Tensor N D) (scale : ℝ) (i : Fin N)
    (first : KeyBlock N) (rest : List (KeyBlock N))
    (hsep : Separated first.keys rest)
    (hcover : collected first.keys rest = causalSet i) :
    onlineFold (score Q K scale i) V
      (onlineFirst (score Q K scale i) V first.keys first.nonempty) rest =
      ⟨savedMax Q K scale i, savedNormalizer Q K scale i, savedOutput Q K V scale i⟩ := by
  rw [onlineFirst_eq_batchRow, onlineFold_eq_batchRow _ _ _ _ _ hsep]
  have h := causal_batchRow Q K V scale i
  simpa only [hcover] using h

#axiomsClean blockMass_rescale
#axiomsClean blockNumerator_rescale
#axiomsClean onlineFirst_eq_batchRow
#axiomsClean onlineStep_eq_batchRow
#axiomsClean onlineFold_eq_batchRow
#axiomsClean online_forward_eq_saved
#specNonCircular batchRow avoiding [onlineFirst, onlineStep, onlineFold]

end VeriTile.Bench.Repairs.TritonAttention.Whole001.Math
