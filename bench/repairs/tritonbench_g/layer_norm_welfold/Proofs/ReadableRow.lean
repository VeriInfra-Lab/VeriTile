import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.ReadableNormalize
namespace VeriTile.Bench.Repairs.LayerNormWelfold.Readable
open VeriTile.Triton FP.Structural Reuse
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false
theorem whole_row {α : Type} [Inhabited α] (M : Algebra α) (eq : α→α→Bool)
    (s : State α) (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : ReductionTree B) (reuse : Bool) (hB : 0<B) (sep : Separation X W Bias Y Mean Rstd) :
    ∃ final, FP.Structural.exec (withEquality M eq) (SourceCode.selected reuse X W Bias Y Mean Rstd S D B (Readable.tree tree)) s=some final ∧
      RowEffect M s final X W Bias Y Mean Rstd D B tree := by
  obtain ⟨u,hu,hs⟩ := stats_run (withEquality M eq) s X Mean Rstd S D B tree (sep.inputs X (.inl rfl)).1
  rw [mean_equality,inverse_equality] at hs
  obtain ⟨t,ht,hreg,hout,hframe⟩ := normalize_whole_row M eq u X W Bias Y S D B (s.pids 0) tree reuse
    _ _ hB (sep.inputs X (.inl rfl)).2.2 (sep.inputs W (.inr (.inl rfl))).2.2
    (sep.inputs Bias (.inr (.inr rfl))).2.2 hs.mean hs.inverse hs.row
  have hinput : ∀ q, q=X ∨ q=W ∨ q=Bias → u.mem q=s.mem q := by
    intro q hq
    funext off
    rw [hs.mem]
    exact statistics_frame s Mean Rstd _ _ q off (.inl (sep.inputs q hq).1) (.inl (sep.inputs q hq).2.1)
  refine ⟨t,?_,?_⟩
  · change run (withEquality M eq)
      (SourceCode.selected reuse X W Bias Y Mean Rstd S D B (Readable.tree tree)).surfaceBody s=some t
    rw [source_split,run_append,hu]
    exact ht
  · refine ⟨hreg.pids.trans hs.pids,hreg.undef.trans hs.undef,?_,?_,?_,?_⟩
    · rw [hframe Mean (s.pids 0) (.inl sep.mean_y),hs.mem]
      simp [State.write,sep.mean_rstd]
    · rw [hframe Rstd (s.pids 0) (.inl sep.rstd_y),hs.mem]
      simp [State.write]
    · intro col hc
      simpa only [normalizeValue,hinput X (.inl rfl),hinput W (.inr (.inl rfl)),
        hinput Bias (.inr (.inr rfl))] using hout col hc
    · intro q off hm
      rw [hframe q off hm.2.2,hs.mem]
      exact statistics_frame s Mean Rstd _ _ q off hm.1 hm.2.1


#axiomsClean whole_row
end
end VeriTile.Bench.Repairs.LayerNormWelfold.Readable
