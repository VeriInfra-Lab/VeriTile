import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_SafetyPrefix

/-! Safety follows the actual successful prefix states, with every access
bound obtained from the declared row extents. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.Safety
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

theorem append_after (bounds : RegionBounds) (first rest : List Stmt) (s t : BlockState)
    (hs : Stmt.TraceSafeList bounds first s) (he : stepStmts first s = some t)
    (ht : Stmt.TraceSafeList bounds rest t) :
    Stmt.TraceSafeList bounds (first ++ rest) s := by
  apply TraceProofs.append_safe first s hs
  intro u hu
  have : u = t := Option.some.inj (hu.symm.trans he)
  simpa only [this] using ht

theorem firstStream_safe (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block)
    (u : BlockState) (bounds : RegionBounds)
    (hb : FlatDomain.RowBounds f r stride M N u.pid bounds) :
    Stmt.TraceSafeList bounds ((Execution.source f r stride M N eps p block hBlock).body.take 12) u := by
  rw [Execution.source_firstStreamPrefix, List.append_assoc]
  apply append_after bounds _ _ u (Execution.loadState f r stride N block u)
  · exact loadPrefix_safe f r stride N block u bounds hb.x
  · exact Execution.loadPrefix_exec f r stride N block u
  apply append_after bounds _ _ _ (Execution.firstScaledState f r stride N block u)
  · exact scale_safe f.rowscale r.rowscale u.pid block _ bounds
      (Execution.loadState_row f r stride N block u) hb.rowscale
  · apply Execution.scalePhase_exec
    · exact Execution.loadState_row f r stride N block u
    · rw [Execution.loadState_x, Execution.loaded_realTile]
  apply dropout_safe f.dropout f.dropoutMask r.seeds r.dropoutMask u.pid N block p _ bounds
  · rw [Execution.firstScaledState, Execution.scaleState_row]
    exact Execution.loadState_row f r stride N block u
  · rw [Execution.firstScaledState, Execution.scaleState_cols]
    exact Execution.loadState_cols f r stride N block u
  · exact hb.seeds
  · exact hb.mask

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.Safety
