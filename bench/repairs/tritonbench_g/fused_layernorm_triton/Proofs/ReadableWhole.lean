import bench.repairs.tritonbench_g.fused_layernorm_triton.Proofs.ReadableNormalizeLoop
import bench.tritonbench_g.fused_layernorm_triton.Proofs.WholeGrid

namespace VeriTile.Bench.Repairs.FusedLayernormTriton.ReadableWhole
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.FusedLayernormTriton FPStages Whole
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def statisticsCode (hoist : Bool) (X W Bias Y Mean Rstd : RegionName)
    (S D B : Nat) (tree : Kernels.ReductionTree B) : List ComputeStmt :=
  (SourcePieces.statistics hoist X W Bias Y Mean Rstd S D B tree).before ++
  [.forRange "roffset" B D B (SourcePieces.statistics hoist X W Bias Y Mean Rstd S D B tree).body] ++
  (SourcePieces.normalization hoist X W Bias Y Mean Rstd S D B tree).before

theorem statistics_ready (hoist : Bool) (M : Algebra α) (eq32 : α → α → Bool)
    (s : State α) (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : Kernels.ReductionTree B) (hB : 0<B) :
    ∃ final,run (withEquality M eq32) (statisticsCode hoist X W Bias Y Mean Rstd S D B tree) s=some final ∧
      StatisticsReady s final Mean Rstd D M
        (treeValue M eq32 (allLaneValues M s X D B) (ReadableStatistics.legacyTree tree)) ∧
      (hoist=Bool.true → ReadableNormalize.Pins final (s.pids 0)
        (treeValue M eq32 (allLaneValues M s X D B) (ReadableStatistics.legacyTree tree)).mean
        (rstdValue M D (treeValue M eq32 (allLaneValues M s X D B) (ReadableStatistics.legacyTree tree)))) := by
  obtain ⟨u,hu,hs⟩ := ReadableStatistics.statistics_ready M eq32 s X W Bias Y Mean Rstd S D B tree hB
  change run (withEquality M eq32) (statisticsCode Bool.false X W Bias Y Mean Rstd S D B tree) s=some u at hu
  cases hoist
  · exact ⟨u,hu,hs,by simp⟩
  · obtain ⟨t,ht,hmem,hreg,hpin⟩ := ReadableNormalize.projections_run M eq32 u X W Bias Y Mean Rstd
      S D B (s.pids 0) tree _ _ hs.mean hs.rstd hs.row
    refine ⟨t,?_,⟨hmem.trans hs.mem,hreg.pids.trans hs.pids,hreg.undef.trans hs.undef,
      hreg.mean,hreg.rstd,hreg.row⟩,fun _ => hpin⟩
    have heq : statisticsCode Bool.true X W Bias Y Mean Rstd S D B tree =
        statisticsCode Bool.false X W Bias Y Mean Rstd S D B tree ++
          SourcePieces.projections X W Bias Y Mean Rstd S D B tree := by
      simp only [statisticsCode,SourcePieces.common_prefix,SourcePieces.common_body,
        SourcePieces.hoisted_before,List.append_assoc]
    rw [heq,run_append,hu]
    exact ht

theorem whole_row (hoist : Bool) (M : Algebra α) (eq32 : α → α → Bool)
    (s : State α) (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : Kernels.ReductionTree B) (hB : 0<B) (sep : Separation X W Bias Y Mean Rstd) :
    ∃ final,FP.Structural.exec (withEquality M eq32)
      (SourcePieces.source hoist X W Bias Y Mean Rstd S D B tree) s=some final ∧
      RowEffect M eq32 s final X W Bias Y Mean Rstd D B (ReadableStatistics.legacyTree tree) := by
  obtain ⟨u,hu,hs,hpin⟩ := statistics_ready hoist M eq32 s X W Bias Y Mean Rstd S D B tree hB
  obtain ⟨t,ht,hreg,hout,hframe⟩ := ReadableNormalizeLoop.whole_row hoist M eq32 u
    X W Bias Y Mean Rstd S D B (s.pids 0) tree _ _ hB
    (sep.inputs X (.inl rfl)).2.2 (sep.inputs W (.inr (.inl rfl))).2.2
    (sep.inputs Bias (.inr (.inr rfl))).2.2 hs.mean hs.rstd hs.row hpin
  have hinput : ∀ q,q=X ∨ q=W ∨ q=Bias → u.mem q=s.mem q := by
    intro q hq
    funext off
    rw [hs.mem]
    exact statistics_frame M s Mean Rstd D _ q off (.inl (sep.inputs q hq).1) (.inl (sep.inputs q hq).2.1)
  refine ⟨t,?_,?_,?_,?_,?_,?_,?_⟩
  · change run _ (SourcePieces.source hoist X W Bias Y Mean Rstd S D B tree).surfaceBody s=some t
    rw [SourcePieces.complete]
    change run _ (statisticsCode hoist X W Bias Y Mean Rstd S D B tree ++ _) s=some t
    simp only [run_append,hu,Option.bind,run,step,ht,bind,pure]
  · exact hreg.pids.trans hs.pids
  · exact hreg.undef.trans hs.undef
  · rw [hframe Mean (s.pids 0) (.inl sep.mean_y),hs.mem]
    simp [statisticsMemory,State.write,sep.mean_rstd,rowValue]
  · rw [hframe Rstd (s.pids 0) (.inl sep.rstd_y),hs.mem]
    simp [statisticsMemory,rowValue]
  · intro col hc
    have hv := hout col hc
    simpa only [normalizeValue,hinput X (.inl rfl),hinput W (.inr (.inl rfl)),
      hinput Bias (.inr (.inr rfl)),rowValue] using hv
  · intro q off hm
    rw [hframe q off hm.2.2,hs.mem]
    exact statistics_frame M s Mean Rstd D _ q off hm.1 hm.2.1

theorem whole_grid (hoist : Bool) (M : Algebra α)
    (eq32 : α → α → Bool) (s : State α) (X W Bias Y Mean Rstd : RegionName)
    (S D B : Nat) (tree : Kernels.ReductionTree B) (hB : 0 < B)
    (sep : Separation X W Bias Y Mean Rstd) (rows : List Nat) :
    ∃ final, runRows (withEquality M eq32) (SourcePieces.source hoist X W Bias Y Mean Rstd S D B tree) rows s = some final ∧
      GridEffect M eq32 s final X W Bias Y Mean Rstd D B (ReadableStatistics.legacyTree tree) rows := by
  induction rows generalizing s with
  | nil =>
    refine ⟨s, rfl, ?_, ?_, ?_, ?_⟩
    · simp
    · simp
    · simp
    · intro q off h; rfl
  | cons row rows ih =>
    obtain ⟨u, hu, he⟩ := whole_row hoist M eq32 (atRow s row) X W Bias Y Mean Rstd S D B tree hB sep
    obtain ⟨t, ht, hf⟩ := ih u
    have hinput : ∀ q, q = X ∨ q = W ∨ q = Bias → u.mem q = s.mem q := by
      intro q hq
      funext off
      exact he.frame q off ⟨.inl (sep.inputs q hq).1, .inl (sep.inputs q hq).2.1, .inl (sep.inputs q hq).2.2⟩
    have hv : ∀ r, rowValue M eq32 (atRow u r) X D B (ReadableStatistics.legacyTree tree) = rowValue M eq32 (atRow s r) X D B (ReadableStatistics.legacyTree tree) :=
      fun r => row_value_congr M eq32 u s X D B r (ReadableStatistics.legacyTree tree) (hinput X (.inl rfl))
    have ho : ∀ r col, rowOutput M eq32 u X W Bias D B r col (ReadableStatistics.legacyTree tree) = rowOutput M eq32 s X W Bias D B r col (ReadableStatistics.legacyTree tree) := by
      intro r col
      simp only [rowOutput, hv, normalizeValue, hinput X (.inl rfl),
        hinput W (.inr (.inl rfl)), hinput Bias (.inr (.inr rfl))]
    refine ⟨t, by simp [runRows, hu, ht], ?_, ?_, ?_, ?_⟩
    · intro r hr
      by_cases hin : r ∈ rows
      · simpa only [hv] using hf.mean r hin
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


#axiomsClean whole_row
#axiomsClean whole_grid
end
end VeriTile.Bench.Repairs.FusedLayernormTriton.ReadableWhole
