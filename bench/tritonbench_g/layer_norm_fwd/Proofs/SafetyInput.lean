import bench.tritonbench_g.layer_norm_fwd.Proofs.SafetyFirstStream

/-! Finite safety for the complete second input and residual subtree. Bounds
for optional arguments are used only after their actual flag is true. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.Safety
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

theorem secondBody_safe (f : Flags) (r : Regions) (M row start N block : Nat)
    (p : ℝ) (s : BlockState) (bounds : RegionBounds) (hBlock : Random.ArangeSizeDomain block)
    (hptr : s.regs .ptr [] "X1" = some (Tile.scalar (r.x1, start)))
    (hrow : s.regs .nat [] "row" = some (Tile.scalar row))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hx : ∀ col, col < N → start+col < bounds r.x1)
    (hscale : f.rowscale = Bool.true → M+row < bounds r.rowscale)
    (hseed : f.dropout = Bool.true → M+row < bounds r.seeds)
    (hmask : f.dropout = Bool.true → f.dropoutMask = Bool.true →
      ∀ col, col < N → (M+row)*N+col < bounds r.dropoutMask) :
    Stmt.TraceSafeList bounds (InputSuffix.secondBody f r M N block p) s := by
  simp only [InputSuffix.secondBody, List.append_assoc]
  apply append_after bounds _ _ s (InputSuffix.loadedState r start N block s)
  · rw [singleton]
    simpa only [Stmt.TraceSafe] using rowLoadOp_safe "X1" r.x1 start N block s bounds hptr hcols hx
  · rw [InputSuffix.exec_single]
    exact stepStmt_assign_eq_some (InputSuffix.rowLoadOp_eval "X1" r.x1 start N block s hptr hcols)
  apply append_after bounds _ _ _ (InputSuffix.scaledState f r M row start N block s)
  · apply secondScale_safe f.rowscale r.rowscale M row block _ bounds _ hscale
    simpa [InputSuffix.loadedState] using hrow
  · apply InputSuffix.scalePhase_exec
    · simpa [InputSuffix.loadedState] using hrow
    · simp [InputSuffix.loadedState]
  apply TraceProofs.append_safe
  · apply secondDropout_safe f.dropout f.dropoutMask r.seeds r.dropoutMask M row N block p _ bounds
    · cases hf : f.rowscale <;>
        simpa [InputSuffix.scaledState, InputSuffix.scaleState, InputSuffix.loadedState, hf] using hrow
    · cases hf : f.rowscale <;>
        simpa [InputSuffix.scaledState, InputSuffix.scaleState, InputSuffix.loadedState, hf] using hcols
    · exact hseed
    · exact hmask
  intro t _
  apply TraceProofs.all_safe
  simp [InputSuffix.addStmt, Stmt.TraceSafe, Op.SafeAt]

theorem residual_safe (enabled : Bool) (region : RegionName) (start N block : Nat)
    (s : BlockState) (bounds : RegionBounds)
    (hptr : enabled = Bool.true → s.regs .ptr [] "RESIDUAL" = some (Tile.scalar (region, start)))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hb : enabled = Bool.true → ∀ col, col < N → start+col < bounds region) :
    Stmt.TraceSafe bounds (InputSuffix.residualStmt enabled N block) s := by
  rw [InputSuffix.residualStmt, if_const]
  intro he
  apply TraceProofs.assign_cons (rowLoadOp_safe "RESIDUAL" region start N block s bounds
    (hptr he) hcols (hb he))
  intro value
  apply TraceProofs.all_safe
  simp [InputSuffix.addStmt, Stmt.TraceSafe, Op.SafeAt]

theorem residualOut_safe (enabled : Bool) (region : RegionName) (start N block : Nat)
    (s : BlockState) (bounds : RegionBounds)
    (hptr : enabled = Bool.true → s.regs .ptr [] "RESIDUAL_OUT" = some (Tile.scalar (region, start)))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hb : enabled = Bool.true → ∀ col, col < N → start+col < bounds region) :
    Stmt.TraceSafe bounds (InputSuffix.residualOutStmt enabled N block) s := by
  rw [InputSuffix.residualOutStmt, if_const]
  intro he
  rw [singleton]
  exact rowStore_safe "RESIDUAL_OUT" "x" region start N block s bounds (hptr he) hcols (hb he)

theorem inputPhase_safe (f : Flags) (r : Regions) (stride : Strides)
    (M N block : Nat) (p : ℝ) (u : BlockState) (bounds : RegionBounds)
    (hBlock : Random.ArangeSizeDomain block) (hN : N ≤ block)
    (hsep : Domain.ReadWriteSeparated f r)
    (hb : FlatDomain.RowBounds f r stride M N u.pid bounds) :
    Stmt.TraceSafeList bounds (InputSuffix.phase f r M N block p)
      (Execution.firstStreamState f r stride N block p u) := by
  have hp := Execution.loadState_pointers f r stride N block u
  have hrptr : f.residual = Bool.true →
      (InputPrefix.secondState f r stride M N block p u).regs .ptr [] "RESIDUAL" =
        some (Tile.scalar (r.residual, u.pid * stride.residual)) := by
    intro hf
    rw [InputPrefix.secondState_regs _ _ _ _ _ _ _ _ _ _ _
      (by decide) (by decide) (by decide) (by decide)]
    exact hp.2.2.1 hf
  have hrcols : (InputPrefix.secondState f r stride M N block p u).regs .nat [block] "cols" =
      some (Tile.vec (fun j => j.val)) := by
    rw [InputPrefix.secondState_regs _ _ _ _ _ _ _ _ _ _ _
      (by decide) (by decide) (by decide) (by decide)]
    exact Execution.loadState_cols f r stride N block u
  have he := InputSuffix.residualStmt_exec f.residual r.residual (u.pid*stride.residual) N block
    (InputPrefix.twoValues f r stride M N block p u) (InputPrefix.secondState f r stride M N block p u)
    hrptr hrcols (InputPrefix.secondState_x f r stride M N block p u hN hsep)
  unfold InputSuffix.phase
  rw [TraceProofs.cons_of_run_iff (InputPrefix.secondStmt_exec f r stride M N block p u hBlock)]
  constructor
  · rw [if_const]
    intro hf
    apply secondBody_safe f r M u.pid (u.pid*stride.x1) N block p _ bounds hBlock
    · rw [Execution.firstStreamState_regs _ _ _ _ _ _ _ _ _ _
        (by decide) (by decide) (by decide)]
      exact hp.2.1 hf
    · exact Execution.firstStreamState_row f r stride N block p u
    · exact Execution.firstStreamState_cols f r stride N block p u
    · exact hb.x1 hf
    · exact hb.rowscale1 hf
    · exact hb.seeds1 hf
    · exact hb.mask1 hf
  rw [TraceProofs.cons_of_run_iff he]
  constructor
  · exact residual_safe f.residual r.residual (u.pid*stride.residual) N block _ bounds
      hrptr hrcols hb.residual
  rw [singleton]
  apply residualOut_safe f.residualOut r.residualOut (u.pid*stride.residualOut) N block _ bounds
  · intro hf
    change (InputPrefix.afterResidual f r stride M N block p u).regs .ptr [] "RESIDUAL_OUT" = _
    rw [InputPrefix.afterResidual_regs _ _ _ _ _ _ _ _ _ _ _
      (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hp.2.2.2.1 hf
  · change (InputPrefix.afterResidual f r stride M N block p u).regs .nat [block] "cols" = _
    rw [InputPrefix.afterResidual_regs _ _ _ _ _ _ _ _ _ _ _
      (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact Execution.loadState_cols f r stride N block u
  · exact hb.residualOut

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.Safety
