import bench.tritonbench_g.seeded_dropout.Proofs.Execution
import bench.tritonbench_g.seeded_dropout.Proofs.FlatMemory
namespace VeriTile.Bench.TritonBenchG.SeededDropout.Safety
open VeriTile.Triton TypedSource Execution
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 3000

theorem safe_append (bounds : RegionBounds) (before after : List Stmt) (s : BlockState) :
    Stmt.TraceSafeList bounds (before ++ after) s ↔
      Stmt.TraceSafeList bounds before s ∧
        ∀ t, stepStmts before s = some t → Stmt.TraceSafeList bounds after t := by
  induction before generalizing s with
  | nil => simp [Stmt.TraceSafeList]
  | cons st rest ih =>
    cases he : stepStmt st s <;> simp [Stmt.TraceSafeList, stepStmts, he, ih, and_assoc]

theorem prefix_safe (kind : Random.IntegerType) (x y : RegionName) (N : Nat)
    (p : ℝ) (seed : SeedCarrier kind) (B : Nat) (u : BlockState) (bounds : RegionBounds)
    (hx : ∀ j : Fin B, Index.offset u.pid B j < N → Index.offset u.pid B j < bounds x) :
    Stmt.TraceSafeList bounds ((Algorithm.kernel kind Bool.false x y N p seed B).body.take 7) u := by
  cases kind
  all_goals simp [Algorithm.kernel, Algorithm.seedStmt, Stmt.TraceSafeList, Stmt.TraceSafe,
    Op.SafeAt, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul, ComparableDType.lt,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, BlockState.setReg]
  all_goals exact fun j hj => hx j hj

theorem suffix_safe (kind : Random.IntegerType) (x y : RegionName) (N : Nat)
    (p : ℝ) (seed : SeedCarrier kind) (B : Nat) (hB : SourceDomain.LegalBlock B)
    (u : BlockState) (hN : N < 2^31) (hpid : u.pid < SourceDomain.grid N B)
    (hseed : kind.inRange (seedInt kind seed) = Bool.true) (bounds : RegionBounds)
    (hy : ∀ j : Fin B, Index.offset u.pid B j < N → Index.offset u.pid B j < bounds y) :
    Stmt.TraceSafeList bounds ((Algorithm.kernel kind Bool.false x y N p seed B).body.drop 7)
      (prefixState kind x N seed B u) := by
  have hr := prefix_rand kind x N seed B hB u hN hpid hseed
  simp only [Algorithm.kernel, List.drop_succ_cons, List.drop_zero,
    Stmt.TraceSafeList, Stmt.TraceSafe, stepStmt]
  rw [hr]
  cases kind
  all_goals simp [Algorithm.seedOp, Op.SafeAt, MaskOpt.SafeAt,
    evalOp.eq_def, prefix_x, prefix_offsets, prefix_mask,
    Tile.bop, Tile.cop, Tile.select, NumericDType.sub, NumericDType.div,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MemAccess.SafeAt,
    MaskOpt.Active]
  all_goals exact fun j hj => hy j hj

theorem trace_safe (kind : Random.IntegerType) (x y : RegionName) (N : Nat)
    (p : ℝ) (seed : SeedCarrier kind) (B : Nat) (hB : SourceDomain.LegalBlock B)
    (u : BlockState) (hN : N < 2^31) (hpid : u.pid < SourceDomain.grid N B)
    (hseed : kind.inRange (seedInt kind seed) = Bool.true) (bounds : RegionBounds)
    (hb : FlatMemory.Bounds x y N B u bounds) :
    (original kind x y N p seed B hB).TraceSafe bounds u := by
  rw [original, Algorithm.source_eq]
  unfold Kernel.TraceSafe
  rw [← List.take_append_drop 7 (Algorithm.kernel kind Bool.false x y N p seed B).body,
    safe_append]
  refine ⟨prefix_safe kind x y N p seed B u bounds hb.input, ?_⟩
  intro t ht
  rw [prefix_exec] at ht
  cases ht
  exact suffix_safe kind x y N p seed B hB u hN hpid hseed bounds hb.output

theorem flatten_ok (kind : Random.IntegerType) (x y : RegionName) (N : Nat)
    (p : ℝ) (seed : SeedCarrier kind) (B : Nat) (hB : SourceDomain.LegalBlock B) :
    (original kind x y N p seed B hB).FlattenOk := by
  rw [original, Algorithm.source_eq]
  cases kind
  all_goals simp [Kernel.FlattenOk, Algorithm.kernel, Algorithm.seedStmt, Algorithm.seedOp,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

end
end VeriTile.Bench.TritonBenchG.SeededDropout.Safety
