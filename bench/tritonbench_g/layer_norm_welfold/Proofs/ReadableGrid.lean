import bench.tritonbench_g.layer_norm_welfold.Proofs.ReadableRow
import bench.tritonbench_g.layer_norm_welfold.Proofs.Grid
namespace VeriTile.Bench.Repairs.LayerNormWelfold.Readable
open VeriTile.Triton FP.Structural Reuse
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false
theorem whole_grid {α : Type} [Inhabited α] (M : Algebra α)
    (eq32 : α → α → Bool) (s : State α) (X W Bias Y Mean Rstd : RegionName)
    (S D B : Nat) (tree : ReductionTree B) (reuse : Bool) (hB : 0 < B)
    (sep : Separation X W Bias Y Mean Rstd) (rows : List Nat) :
    ∃ final, SourceCode.runRows (withEquality M eq32) (SourceCode.selected reuse X W Bias Y Mean Rstd S D B (Readable.tree tree)) rows s = some final ∧
      GridEffect M s final X W Bias Y Mean Rstd D B tree rows := by
  induction rows generalizing s with
  | nil =>
    refine ⟨s, rfl, ?_, ?_, ?_, ?_⟩
    · simp
    · simp
    · simp
    · intro q off h; rfl
  | cons row rows ih =>
    obtain ⟨u, hu, he⟩ := whole_row M eq32 (atRow s row) X W Bias Y Mean Rstd S D B tree reuse hB sep
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
    refine ⟨t, ?_, ?_, ?_, ?_, ?_⟩
    · change (exec (withEquality M eq32)
        (SourceCode.selected reuse X W Bias Y Mean Rstd S D B (Readable.tree tree)) (atRow s row)).bind _=some t
      rw [hu]
      exact ht
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
end VeriTile.Bench.Repairs.LayerNormWelfold.Readable
