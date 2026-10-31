import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_SafetyStreams
import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_Real

/-! Both actual pointer-building prefixes are safe. The sole memory access is
the original masked X load; inactive columns impose no backing-memory bound. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.Safety
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

theorem loadPrefix_safe (f : Flags) (r : Regions) (stride : Strides) (N block : Nat)
    (s : BlockState) (bounds : RegionBounds)
    (hb : ∀ col, col < N → s.pid*stride.x+col < bounds r.x) :
    Stmt.TraceSafeList bounds (Execution.loadPrefix f r stride N block) s := by
  rcases f with ⟨rms, residual, residualOut, bias, dropout, dropoutMask, rowscale, x1, w1, b1⟩
  cases residual <;> cases residualOut <;> cases x1 <;> cases w1
  all_goals
    simp [Execution.loadPrefix, Execution.baseStmt, Stmt.TraceSafeList, Stmt.TraceSafe,
      Op.SafeAt, MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
      Op.PointerAddressesSafeOn, Op.MemorySafe, MaskOpt.Active,
      stepStmts, stepStmt, evalOp, Tile.bop, Tile.cop, Tile.ptrAdd,
      NumericDType.mul, ComparableDType.lt]
  all_goals exact fun j hj => hb j.val hj

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.Safety
