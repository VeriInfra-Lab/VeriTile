import bench.tritonbench_g.reversed_cumsum.Proofs.SourceAudit
import VeriTile.Triton.Float.Structural

/-! Whole typed executions are equal for every algebra, input/output storage
profile, initial state and shape. The only rewrite is discrete comparison.
Existence of successful executions is a separate obligation. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsum.FPProof
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

theorem prefix_run (dtype out_dtype : FloatDType) (s z : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat)
    {α : Type} [Inhabited α] (A : FP.Structural.Algebra α) (u : FP.Structural.State α) :
    FP.Structural.run A
      ((TypedSource.original dtype out_dtype s z s_s_h s_s_t s_s_d T S BT BS).surfaceBody.take 4) u =
    FP.Structural.run A
      ((TypedSource.compareRight dtype out_dtype s z s_s_h s_s_t s_s_d T S BT BS).surfaceBody.take 4) u := by
  simp [TypedSource.original, TypedSource.compareRight, ComputeKernel.surfaceBody,
    FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr,
    FP.Structural.evalOp_unfold, FP.Structural.numericLe, FP.Structural.natLe,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    TileShape.dropInsertedIndex, bind, Option.bind]

theorem whole_state (dtype out_dtype : FloatDType) (s z : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat)
    {α : Type} [Inhabited α] (A : FP.Structural.Algebra α) (u : FP.Structural.State α) :
    FP.Structural.exec A (TypedSource.original dtype out_dtype s z s_s_h s_s_t s_s_d T S BT BS) u =
    FP.Structural.exec A (TypedSource.compareRight dtype out_dtype s z s_s_h s_s_t s_s_d T S BT BS) u := by
  change FP.Structural.run A
    (TypedSource.original dtype out_dtype s z s_s_h s_s_t s_s_d T S BT BS).surfaceBody u =
    FP.Structural.run A
    (TypedSource.compareRight dtype out_dtype s z s_s_h s_s_t s_s_d T S BT BS).surfaceBody u
  conv_lhs => rw [← List.take_append_drop 4
    (TypedSource.original dtype out_dtype s z s_s_h s_s_t s_s_d T S BT BS).surfaceBody]
  conv_rhs => rw [← List.take_append_drop 4
    (TypedSource.compareRight dtype out_dtype s z s_s_h s_s_t s_s_d T S BT BS).surfaceBody]
  simp only [FP.Structural.run_append, prefix_run, SourceAudit.common_suffix]

#axiomsClean prefix_run
#axiomsClean whole_state
end
end VeriTile.Bench.TritonBenchG.ReversedCumsum.FPProof
