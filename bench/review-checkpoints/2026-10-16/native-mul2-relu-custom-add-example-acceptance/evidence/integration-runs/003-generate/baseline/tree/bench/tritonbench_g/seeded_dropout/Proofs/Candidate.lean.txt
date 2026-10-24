import bench.tritonbench_g.seeded_dropout.Proofs.Execution
namespace VeriTile.Bench.TritonBenchG.SeededDropout.Candidate
open VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 3000

theorem prefix_flat (A : FlatAlloc) (kind : Random.IntegerType) (x y : RegionName)
    (N : Nat) (p : ℝ) (seed : SeedCarrier kind) (B : Nat) (u : BlockState) :
    stepStmts (A.flattenStmts ((Algorithm.kernel kind Bool.false x y N p seed B).body.take 4)) u =
      stepStmts (A.flattenStmts ((Algorithm.kernel kind Bool.true x y N p seed B).body.take 4)) u := by
  cases kind
  all_goals simp [Algorithm.kernel, Algorithm.seedStmt,
    FlatAlloc.flattenStmts, FlatAlloc.flattenStmt, FlatAlloc.flattenOp,
    stepStmts, stepStmt, evalOp.eq_def, Tile.bop, NumericDType.mul, Nat.mul_comm]

theorem suffix_eq (kind : Random.IntegerType) (x y : RegionName)
    (N : Nat) (p : ℝ) (seed : SeedCarrier kind) (B : Nat) :
    (Algorithm.kernel kind Bool.false x y N p seed B).body.drop 4 =
      (Algorithm.kernel kind Bool.true x y N p seed B).body.drop 4 := rfl

theorem flattenStmts_append (A : FlatAlloc) (before after : List Stmt) :
    A.flattenStmts (before ++ after) = A.flattenStmts before ++ A.flattenStmts after := by
  induction before with
  | nil => rfl
  | cons s rest ih => simp only [List.cons_append, FlatAlloc.flattenStmts, ih]

theorem algorithm_flat_exec (A : FlatAlloc) (kind : Random.IntegerType) (x y : RegionName)
    (N : Nat) (p : ℝ) (seed : SeedCarrier kind) (B : Nat) (u : BlockState) :
    exec (A.flattenKernel (Algorithm.kernel kind Bool.false x y N p seed B)) u =
      exec (A.flattenKernel (Algorithm.kernel kind Bool.true x y N p seed B)) u := by
  unfold exec FlatAlloc.flattenKernel
  conv_lhs => rw [← List.take_append_drop 4 (Algorithm.kernel kind Bool.false x y N p seed B).body]
  conv_rhs => rw [← List.take_append_drop 4 (Algorithm.kernel kind Bool.true x y N p seed B).body]
  simp only [flattenStmts_append, Execution.stepStmts_append, prefix_flat, suffix_eq]

/-- Equality includes failures and every field of the resulting state. Only
integer index multiplication commutes; all floating/RNG source is identical. -/
theorem flat_exec (A : FlatAlloc) (storage : Storage) (kind : Random.IntegerType)
    (x y : RegionName) (N : Nat) (p : ℝ) (seed : SeedCarrier kind)
    (B : Nat) (hB : SourceDomain.LegalBlock B) (u : BlockState) :
    exec (A.flattenKernel (source storage kind Bool.false x y N p seed B hB).eraseDType.toAlgKernel) u =
      exec (A.flattenKernel (source storage kind Bool.true x y N p seed B hB).eraseDType.toAlgKernel) u := by
  rw [Erasure.kernel, Erasure.kernel, Algorithm.source_eq, Algorithm.source_eq]
  exact algorithm_flat_exec A kind x y N p seed B u

end
end VeriTile.Bench.TritonBenchG.SeededDropout.Candidate
