import bench.tritonbench_g.layer_norm_fwd.Proofs.SafetyPrimitives

/-! Address safety of both full scale/dropout streams. Float values stay
abstract, while each actual signed seed load and boolean mask store is bounded.
The proofs do not assume that any numerical expression succeeds. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.Safety
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

theorem scalarLoad_safe (region : RegionName) (row : Nat) (s : BlockState)
    (bounds : RegionBounds)
    (hrow : s.regs .nat [] "row" = some (Tile.scalar row)) (hb : row < bounds region) :
    (Op.load .real (.region region (.ref .nat [] "row")) .none).SafeAt bounds s := by
  simp [Op.SafeAt, MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, evalOp, hrow, hb, Region.cast]

theorem secondScalarLoad_safe (region : RegionName) (M row : Nat) (s : BlockState)
    (bounds : RegionBounds)
    (hrow : s.regs .nat [] "row" = some (Tile.scalar row)) (hb : M+row < bounds region) :
    (Op.load .real (.region region (InputSuffix.secondRowOp M)) .none).SafeAt bounds s := by
  simp [InputSuffix.secondRowOp, Op.SafeAt, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, MaskOpt.Active, evalOp, hrow, hb, Region.cast,
    Tile.bop, NumericDType.add]

theorem scale_safe (enabled : Bool) (region : RegionName) (row block : Nat)
    (s : BlockState) (bounds : RegionBounds)
    (hrow : s.regs .nat [] "row" = some (Tile.scalar row))
    (hb : enabled = Bool.true → row < bounds region) :
    Stmt.TraceSafeList bounds (Execution.scalePhase enabled region block) s := by
  rw [Execution.scalePhase, singleton, if_const]
  intro henabled
  apply TraceProofs.assign_cons (scalarLoad_safe region row s bounds hrow (hb henabled))
  intro value
  apply TraceProofs.all_safe
  simp [Stmt.TraceSafe, Op.SafeAt]

theorem secondScale_safe (enabled : Bool) (region : RegionName) (M row block : Nat)
    (s : BlockState) (bounds : RegionBounds)
    (hrow : s.regs .nat [] "row" = some (Tile.scalar row))
    (hb : enabled = Bool.true → M+row < bounds region) :
    Stmt.TraceSafeList bounds (InputSuffix.scalePhase enabled region M block) s := by
  rw [InputSuffix.scalePhase, singleton, if_const]
  intro henabled
  apply TraceProofs.assign_cons (secondScalarLoad_safe region M row s bounds hrow (hb henabled))
  intro value
  apply TraceProofs.all_safe
  simp [Stmt.TraceSafe, Op.SafeAt]

theorem dropout_safe (enabled storeMask : Bool) (seeds mask : RegionName)
    (row N block : Nat) (p : ℝ) (s : BlockState) (bounds : RegionBounds)
    (hrow : s.regs .nat [] "row" = some (Tile.scalar row))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hseed : enabled = Bool.true → row < bounds seeds)
    (hmask : enabled = Bool.true → storeMask = Bool.true →
      ∀ col, col < N → row*N+col < bounds mask) :
    Stmt.TraceSafeList bounds (Execution.dropoutPhase enabled storeMask seeds mask N block p) s := by
  rw [Execution.dropoutPhase, singleton, if_const]
  intro he
  apply TraceProofs.assign_cons
  · simpa only [Execution.keepOp, Op.SafeAt, and_true] using
      firstRandomOp_safe seeds row block s bounds hrow (hseed he)
  intro keep
  apply TraceProofs.assign_cons
  · simp [Execution.dropoutValueOp, Op.SafeAt]
  intro value
  rw [singleton, if_const]
  intro hm
  rw [singleton]
  apply firstMask_safe mask row N block _ bounds
  · simpa using hrow
  · simpa using hcols
  · exact hmask he hm

theorem secondDropout_safe (enabled storeMask : Bool) (seeds mask : RegionName)
    (M row N block : Nat) (p : ℝ) (s : BlockState) (bounds : RegionBounds)
    (hrow : s.regs .nat [] "row" = some (Tile.scalar row))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hseed : enabled = Bool.true → M+row < bounds seeds)
    (hmask : enabled = Bool.true → storeMask = Bool.true →
      ∀ col, col < N → (M+row)*N+col < bounds mask) :
    Stmt.TraceSafeList bounds (InputSuffix.dropoutPhase enabled storeMask seeds mask M N block p) s := by
  rw [InputSuffix.dropoutPhase, singleton, if_const]
  intro he
  apply TraceProofs.assign_cons
  · simpa only [InputSuffix.secondKeepOp, Op.SafeAt, and_true] using
      secondRandomOp_safe seeds M row block s bounds hrow (hseed he)
  intro keep
  apply TraceProofs.assign_cons
  · simp [InputSuffix.dropoutValueOp, Op.SafeAt]
  intro value
  rw [singleton, if_const]
  intro hm
  rw [singleton]
  apply secondMask_safe mask M row N block _ bounds
  · simpa using hrow
  · simpa using hcols
  · exact hmask he hm

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.Safety
