import bench.tritonbench_g.layer_norm_welfold.Proofs.Row

namespace VeriTile.Bench.Repairs.LayerNormWelfold
open VeriTile.Triton FP.Structural Reuse
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

theorem mean_congr {α : Type} [Inhabited α] (M : Algebra α) (s t : State α)
    (X : RegionName) (D B row : Nat) (tree : ReductionTree B) (hm : s.mem X=t.mem X) :
    meanValue M (atRow s row) X D B tree = meanValue M (atRow t row) X D B tree := by
  unfold meanValue allLanes
  rw [recurrence_congr M Bool.false (atRow s row) (atRow t row) X D B 0 (blockCount D B) _ _ hm rfl]

theorem inverse_congr {α : Type} [Inhabited α] (M : Algebra α) (s t : State α)
    (X : RegionName) (D B row : Nat) (tree : ReductionTree B) (hm : s.mem X=t.mem X) :
    inverseValue M (atRow s row) X D B tree = inverseValue M (atRow t row) X D B tree := by
  unfold inverseValue
  rw [mean_congr M s t X D B row tree hm]
  unfold allLanes
  rw [recurrence_congr M Bool.true (atRow s row) (atRow t row) X D B 0 (blockCount D B) _ _ hm rfl]

def rowOutput {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (X W Bias : RegionName) (D B row col : Nat) (tree : ReductionTree B) : α :=
  normalizeValue M s X W Bias D row col (meanValue M (atRow s row) X D B tree)
    (inverseValue M (atRow s row) X D B tree)

structure GridEffect {α : Type} [Inhabited α] (M : Algebra α)
    (s t : State α) (X W Bias Y Mean Rstd : RegionName) (D B : Nat)
    (tree : ReductionTree B) (rows : List Nat) : Prop where
  mean : ∀ row ∈ rows, t.mem Mean row = .mk .real (meanValue M (atRow s row) X D B tree)
  rstd : ∀ row ∈ rows, t.mem Rstd row = .mk .real (inverseValue M (atRow s row) X D B tree)
  output : ∀ row ∈ rows, ∀ col, col < D → t.mem Y (col + D * row) =
    .mk .bf16 (rowOutput M s X W Bias D B row col tree)
  frame : ∀ q off, (∀ row ∈ rows, rowMiss Y Mean Rstd D row q off) → t.mem q off = s.mem q off

theorem whole_grid {α : Type} [Inhabited α] (M : Algebra α)
    (eq32 : α → α → Bool) (s : State α) (X W Bias Y Mean Rstd : RegionName)
    (S D B : Nat) (tree : ReductionTree B) (hB : 0 < B)
    (sep : Separation X W Bias Y Mean Rstd) (rows : List Nat) :
    ∃ final, runRows (withEquality M eq32) (rowKernel X W Bias Y Mean Rstd S D B tree) rows s = some final ∧
      GridEffect M s final X W Bias Y Mean Rstd D B tree rows := by
  induction rows generalizing s with
  | nil =>
    refine ⟨s, rfl, ?_, ?_, ?_, ?_⟩
    · simp
    · simp
    · simp
    · intro q off h; rfl
  | cons row rows ih =>
    obtain ⟨u, hu, he⟩ := whole_row M eq32 (atRow s row) X W Bias Y Mean Rstd S D B tree hB sep
    obtain ⟨t, ht, hf⟩ := ih u
    have hinput : ∀ q, q = X ∨ q = W ∨ q = Bias → u.mem q = s.mem q := by
      intro q hq
      funext off
      exact he.frame q off ⟨.inl (sep.inputs q hq).1, .inl (sep.inputs q hq).2.1, .inl (sep.inputs q hq).2.2⟩
    have hm : ∀ r, meanValue M (atRow u r) X D B tree = meanValue M (atRow s r) X D B tree :=
      fun r => mean_congr M u s X D B r tree (hinput X (.inl rfl))
    have hv : ∀ r, inverseValue M (atRow u r) X D B tree = inverseValue M (atRow s r) X D B tree :=
      fun r => inverse_congr M u s X D B r tree (hinput X (.inl rfl))
    have ho : ∀ r col, rowOutput M u X W Bias D B r col tree = rowOutput M s X W Bias D B r col tree := by
      intro r col
      simp only [rowOutput, hm, hv, normalizeValue, hinput X (.inl rfl),
        hinput W (.inr (.inl rfl)), hinput Bias (.inr (.inr rfl))]
    refine ⟨t, by simp [runRows, hu, ht], ?_, ?_, ?_, ?_⟩
    · intro r hr
      by_cases hin : r ∈ rows
      · simpa only [hm] using hf.mean r hin
      · have hrw : r = row := by simpa [hin] using hr
        subst r
        rw [hf.frame Mean row ?_]
        · simpa [atRow] using he.mean
        · intro r hr
          exact ⟨.inr (fun hh => hin (hh ▸ hr)), .inl sep.mean_rstd, .inl sep.mean_y⟩
    · intro r hr
      by_cases hin : r ∈ rows
      · simpa only [hv] using hf.rstd r hin
      · have hrw : r = row := by simpa [hin] using hr
        subst r
        rw [hf.frame Rstd row ?_]
        · simpa [atRow] using he.rstd
        · intro r hr
          exact ⟨.inl (Ne.symm sep.mean_rstd), .inr (fun hh => hin (hh ▸ hr)), .inl sep.rstd_y⟩
    · intro r hr col hc
      by_cases hin : r ∈ rows
      · simpa only [ho] using hf.output r hin col hc
      · have hrw : r = row := by simpa [hin] using hr
        subst r
        rw [hf.frame Y (col + D * row) ?_]
        · simpa [rowOutput, atRow, normalizeValue] using he.output col hc
        · intro r hr
          refine ⟨.inl (Ne.symm sep.mean_y), .inl (Ne.symm sep.rstd_y), .inr ?_⟩
          intro j hj heq
          have hrow := (row_column_injective D row r col j hc hj heq).1
          exact hin (hrow ▸ hr)
    · intro q off hmiss
      rw [hf.frame q off (fun r hr => hmiss r (List.mem_cons_of_mem _ hr))]
      exact he.frame q off (by simpa [atRow] using hmiss row (by simp))

#axiomsClean whole_grid
end
end VeriTile.Bench.Repairs.LayerNormWelfold
