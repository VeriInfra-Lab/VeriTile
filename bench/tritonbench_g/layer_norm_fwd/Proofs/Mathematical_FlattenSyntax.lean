import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_FlatDomain
import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_Real

/-! Structural admission to the finite-memory interpreter for every retained
branch of both complete source versions. No source branch is specialized away. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.Safety
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

theorem flatten_append (first rest : List Stmt) :
    StmtList.FlattenOk (first ++ rest) ↔ StmtList.FlattenOk first ∧ StmtList.FlattenOk rest := by
  induction first with
  | nil => simp [StmtList.FlattenOk]
  | cons st first ih => simp [StmtList.FlattenOk, ih, and_assoc]

theorem flatten_drop (code : List Stmt) (count : Nat) (h : StmtList.FlattenOk code) :
    StmtList.FlattenOk (code.drop count) := by
  induction count generalizing code with
  | zero => exact h
  | succ count ih =>
      cases code with
      | nil => trivial
      | cons st rest => exact ih rest h.2

theorem source_body (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block) :
    (Execution.source f r stride M N eps p block hBlock).body =
      ((Execution.loadPrefix f r stride N block ++ Execution.scalePhase f.rowscale r.rowscale block ++
        Execution.dropoutPhase f.dropout f.dropoutMask r.seeds r.dropoutMask N block p) ++
        InputSuffix.phase f r M N block p ++ Normalization.phase f r N block eps) ++
        Affine.phase f r N block := by
  let code := (Execution.source f r stride M N eps p block hBlock).body
  have h15 : code.take 15 = code.take 12 ++ InputSuffix.phase f r M N block p := by
    change code.take (12+3) = _
    rw [List.take_add, InputSuffix.source_phase]
  have h18 : code.take 18 = code.take 15 ++ Normalization.phase f r N block eps := by
    change code.take (15+3) = _
    rw [List.take_add, Normalization.source_phase]
  change code = _
  rw [← List.take_append_drop 18 code, h18, h15]
  rw [Execution.source_firstStreamPrefix, Affine.source_phase]

theorem source_flatten_ok (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block) :
    (Execution.source f r stride M N eps p block hBlock).FlattenOk := by
  rw [Kernel.FlattenOk, source_body]
  simp [flatten_append, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def,
    Execution.loadPrefix, Execution.baseStmt, Execution.scalePhase,
    Execution.dropoutPhase, Execution.keepOp, Execution.randomOp, Execution.dropoutValueOp,
    Execution.dropoutMaskStmt, Op.intToUnsigned,
    InputSuffix.phase, InputSuffix.secondBody, InputSuffix.rowLoadOp, InputSuffix.scalePhase,
    InputSuffix.secondRowOp, InputSuffix.dropoutPhase, InputSuffix.secondKeepOp,
    InputSuffix.secondRandomOp, InputSuffix.dropoutValueOp, InputSuffix.maskStmt,
    InputSuffix.addStmt, InputSuffix.residualStmt, InputSuffix.residualOutStmt,
    Normalization.phase, Normalization.statisticsStmt, Normalization.meanPhase,
    Normalization.meanStmt, Normalization.scalarStore, Normalization.centerVariancePhase,
    Normalization.centerStmt, Normalization.varianceStmt, Normalization.rstdStmt,
    Affine.phase, Affine.maskStmt, Affine.loadPhase, Affine.coefficientOp, Affine.normalizedOp,
    Affine.outputPhase, Affine.valueOp, Affine.storeStmt, Affine.secondStmt]

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.Safety
