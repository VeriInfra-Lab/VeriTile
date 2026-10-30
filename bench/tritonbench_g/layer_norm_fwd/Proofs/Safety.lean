import bench.tritonbench_g.layer_norm_fwd.Proofs.SafetyInput
import bench.tritonbench_g.layer_norm_fwd.Proofs.SafetyNormalization
import bench.tritonbench_g.layer_norm_fwd.Proofs.SafetyAffine

/-! Complete original and candidate trace safety, derived from explicit finite
row bounds and the actual successful phase states. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.Safety
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

theorem inputPrefix_safe (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block)
    (u : BlockState) (hN : N ≤ block) (hsep : Domain.ReadWriteSeparated f r)
    (bounds : RegionBounds) (hb : FlatDomain.RowBounds f r stride M N u.pid bounds) :
    Stmt.TraceSafeList bounds ((Execution.source f r stride M N eps p block hBlock).body.take 15) u := by
  change Stmt.TraceSafeList bounds
    ((Execution.source f r stride M N eps p block hBlock).body.take (12+3)) u
  rw [List.take_add, InputSuffix.source_phase]
  exact append_after bounds _ _ u (Execution.firstStreamState f r stride N block p u)
    (firstStream_safe f r stride M N eps p block hBlock u bounds hb)
    (Execution.source_firstStream_exec f r stride M N eps p block hBlock u)
    (inputPhase_safe f r stride M N block p u bounds hBlock hN hsep hb)

theorem normPrefix_safe (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block)
    (u : BlockState) (hN : N ≤ block) (hsep : Domain.ReadWriteSeparated f r)
    (bounds : RegionBounds) (hb : FlatDomain.RowBounds f r stride M N u.pid bounds) :
    Stmt.TraceSafeList bounds ((Execution.source f r stride M N eps p block hBlock).body.take 18) u := by
  change Stmt.TraceSafeList bounds
    ((Execution.source f r stride M N eps p block hBlock).body.take (15+3)) u
  rw [List.take_add, Normalization.source_phase]
  apply append_after bounds _ _ u (InputPrefix.state f r stride M N block p u)
  · exact inputPrefix_safe f r stride M N eps p block hBlock u hN hsep bounds hb
  · exact InputPrefix.source_exec f r stride M N eps p block hBlock u hN hsep
  apply normalization_safe f (snapshot r stride u) r M N block u.pid eps p hN _ bounds
  · exact InputPrefix.state_x f r stride M N block p u hN hsep
  · rw [InputPrefix.state_regs _ _ _ _ _ _ _ _ _ _ _
      (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact Execution.loadState_cols f r stride N block u
  · rw [InputPrefix.state_regs _ _ _ _ _ _ _ _ _ _ _
      (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact Execution.loadState_row f r stride N block u
  · exact hb.mean
  · exact hb.rstd

theorem source_safe (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block)
    (u : BlockState) (hN : N ≤ block) (hsep : Domain.ReadWriteSeparated f r)
    (bounds : RegionBounds) (hb : FlatDomain.RowBounds f r stride M N u.pid bounds) :
    (Execution.source f r stride M N eps p block hBlock).TraceSafe bounds u := by
  change Stmt.TraceSafeList bounds (Execution.source f r stride M N eps p block hBlock).body u
  rw [← List.take_append_drop 18 (Execution.source f r stride M N eps p block hBlock).body,
    Affine.source_phase]
  apply append_after bounds _ _ u (NormPrefix.state f r stride M N block eps p u)
  · exact normPrefix_safe f r stride M N eps p block hBlock u hN hsep bounds hb
  · exact NormPrefix.source_exec f r stride M N eps p block hBlock u hN hsep
  apply affine_safe f r stride M u.pid N block (InputPrefix.values f r stride M N p u)
    (Formula.mean f (snapshot r stride u) M N p u.pid)
    (Formula.rstd f (snapshot r stride u) M N eps p u.pid) _ bounds
  · rw [NormPrefix.state_regs _ _ _ _ _ _ _ _ _ _ _ _
        (by decide) (by decide) (by decide) (by decide),
      InputPrefix.state_regs _ _ _ _ _ _ _ _ _ _ _
        (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact (Execution.loadState_pointers f r stride N block u).1
  · intro hf
    rw [NormPrefix.state_regs _ _ _ _ _ _ _ _ _ _ _ _
        (by decide) (by decide) (by decide) (by decide),
      InputPrefix.state_regs _ _ _ _ _ _ _ _ _ _ _
        (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact (Execution.loadState_pointers f r stride N block u).2.2.2.2 hf
  · rw [NormPrefix.state_regs _ _ _ _ _ _ _ _ _ _ _ _
        (by decide) (by decide) (by decide) (by decide),
      InputPrefix.state_regs _ _ _ _ _ _ _ _ _ _ _
        (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact Execution.loadState_cols f r stride N block u
  · exact NormPrefix.state_x f r stride M N block eps p u hN hsep
  · exact NormPrefix.state_mean f r stride M N block eps p u
  · exact NormPrefix.state_rstd f r stride M N block eps p u
  · exact hb

theorem append_tail (bounds : RegionBounds) (first rest : List Stmt) (s t : BlockState)
    (h : Stmt.TraceSafeList bounds (first ++ rest) s) (he : stepStmts first s = some t) :
    Stmt.TraceSafeList bounds rest t := by
  induction first generalizing s with
  | nil =>
      simp only [stepStmts] at he
      cases he
      exact h
  | cons st first ih =>
      obtain ⟨v, hv, ht⟩ := TraceProofs.cons_inv he
      simp only [List.cons_append, Stmt.TraceSafeList, hv] at h
      exact ih v h.2 ht

theorem candidate_safe (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block)
    (u : BlockState) (hN : N ≤ block) (hsep : Domain.ReadWriteSeparated f r)
    (bounds : RegionBounds) (hb : FlatDomain.RowBounds f r stride M N u.pid bounds) :
    (CandidateExecution.source f r stride M N eps p block hBlock).TraceSafe bounds u := by
  have hs := source_safe f r stride M N eps p block hBlock u hN hsep bounds hb
  change Stmt.TraceSafeList bounds (Execution.source f r stride M N eps p block hBlock).body u at hs
  have ht : Stmt.TraceSafeList bounds
      ((Execution.source f r stride M N eps p block hBlock).body.drop 10)
      (Execution.loadState f r stride N block u) := by
    apply append_tail bounds ((Execution.source f r stride M N eps p block hBlock).body.take 10)
      _ u (Execution.loadState f r stride N block u)
    · simpa only [List.take_append_drop] using hs
    · exact Execution.source_initial_load f r stride M N eps p block hBlock u
  change Stmt.TraceSafeList bounds (CandidateExecution.source f r stride M N eps p block hBlock).body u
  rw [← List.take_append_drop 10 (CandidateExecution.source f r stride M N eps p block hBlock).body,
    CandidateExecution.source_loadPrefix, CandidateExecution.common_suffix]
  exact append_after bounds _ _ u (Execution.loadState f r stride N block u)
    (candidate_loadPrefix_safe f r stride N block u bounds hb.x)
    (CandidateExecution.loadPrefix_exec f r stride N block u) ht

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.Safety
