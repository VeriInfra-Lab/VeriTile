import bench.tritonbench_g.layer_norm_fwd.Proofs.SafetyStreams

/-! Normalization address safety. Its reductions and arithmetic have no
memory accesses; both actual scalar stores use the established row register. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.Safety
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

theorem centerVariance_safe (rms : Bool) (N block : Nat) (s : BlockState)
    (bounds : RegionBounds) :
    Stmt.TraceSafeList bounds (Normalization.centerVariancePhase rms N block) s := by
  apply TraceProofs.all_safe
  cases rms <;> simp [Normalization.centerVariancePhase, Normalization.centerStmt,
    Normalization.varianceStmt, Stmt.TraceSafe, Op.SafeAt.eq_def]

theorem statistics_safe (rms : Bool) (mean : RegionName) (N block row : Nat)
    (s : BlockState) (bounds : RegionBounds)
    (hrow : s.regs .nat [] "row" = some (Tile.scalar row))
    (hb : rms = Bool.false → row < bounds mean) :
    Stmt.TraceSafe bounds (Normalization.statisticsStmt rms mean N block) s := by
  have hcond : evalOp (.boolNot (.constBool rms)) s = some (Tile.scalar (!rms)) := by
    simp [evalOp, Tile.uop]
  rw [Normalization.statisticsStmt, Stmt.TraceSafe]
  simp only [Op.SafeAt.eq_def, true_and, hcond, Tile.scalar]
  cases hr : rms
  · simp only [Bool.not_false, ↓reduceIte]
    change Stmt.TraceSafeList bounds
      (Normalization.meanStmt N block :: Normalization.scalarStore mean "mean" ::
        Normalization.centerVariancePhase Bool.false N block) s
    apply TraceProofs.assign_cons
    · simp [Normalization.meanStmt, Op.SafeAt.eq_def]
    intro value
    apply Stmt.TraceSafeList.cons_intro
    · exact scalarStore_safe mean "mean" row _ bounds (by simpa using hrow) (hb hr)
    intro t _
    exact centerVariance_safe Bool.false N block t bounds
  · simpa only [Bool.not_true, Bool.false_eq_true, ↓reduceIte] using
      centerVariance_safe Bool.true N block s bounds

theorem normalization_safe (f : Flags) (a : Inputs) (r : Regions)
    (M N block row : Nat) (eps p : ℝ) (hN : N ≤ block) (s : BlockState) (bounds : RegionBounds)
    (hx : s.regs .real [block] "x" = some (Reduction.combinedTile f a M N block p row))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hrow : s.regs .nat [] "row" = some (Tile.scalar row))
    (hmean : f.rms = Bool.false → row < bounds r.mean) (hstd : row < bounds r.rstd) :
    Stmt.TraceSafeList bounds (Normalization.phase f r N block eps) s := by
  unfold Normalization.phase
  rw [TraceProofs.cons_of_run_iff
    (Normalization.statistics_exec f a r.mean M N block p row hN s hx hcols hrow)]
  refine ⟨statistics_safe f.rms r.mean N block row s bounds hrow hmean, ?_⟩
  apply TraceProofs.assign_cons
  · simp [Normalization.rstdStmt, Op.SafeAt]
  intro value
  rw [singleton]
  apply scalarStore_safe r.rstd "rstd" row _ bounds _ hstd
  simpa using (Normalization.statisticsState_row f a r.mean M N block p row s).trans hrow

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.Safety
