import bench.tritonbench_g.seeded_dropout.Proofs.Mathematical_Execution
import bench.tritonbench_g.seeded_dropout.Proofs.Mathematical_FlatMemory
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

end
end VeriTile.Bench.TritonBenchG.SeededDropout.Safety
