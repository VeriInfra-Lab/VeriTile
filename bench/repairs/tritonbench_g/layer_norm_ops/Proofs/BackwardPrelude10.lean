import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardInvariant
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.MemoryHelpers
namespace VeriTile.Bench.Repairs.LayerNormOps.Backward
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits (fold_writes_mem fold_writes_regs)
open Forward (fold_writes_undef vector_bop region_name_eq)
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option pp.maxSteps 1200
set_option linter.unusedSimpArgs false

theorem prelude10 {α : Type} [Inhabited α] (M : Algebra α)
    (p : BackwardParameters) (s : State α)
    (hd : p.hasDres = Bool.true) (hs : p.storeDres = Bool.false)
    :
    ∃ t, run M (prelude p) s = some t ∧ Ready p s t (start p s) (initial M p s) := by
  cases hb : p.hasBias <;> cases hy : p.recompute
  all_goals
    simp [prelude, backwardKernel, backwardSource, ComputeKernel.surfaceBody, run, step, evalExpr,
      evalComputeOp, evalOp_unfold, Option.bind, numeric, bop, integralFloorDiv,
      numericLt, natLt, Region.cast, ofFloat, toFloat, hd, hs, hb, hy]
    constructor <;> simp [initial, literal, loaded, cast, padding, start, stop,
      bop, numeric, hd, hs, hb, hy]
    all_goals try rfl
    all_goals funext i
    all_goals try simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      ofFloat, toFloat, loaded, Forward.cast, Forward.padding, Forward.literal,
      show (0.0 : ℝ) = 0 by norm_num]
    all_goals by_cases hab : (s.pids 0+1)*p.rowsPerProgram < p.rows
    all_goals first
      | rw [if_pos hab, min_eq_left (Nat.le_of_lt hab)]
      | rw [if_neg hab, min_eq_right (Nat.le_of_not_gt hab)]

#axiomsClean prelude10
end
end VeriTile.Bench.Repairs.LayerNormOps.Backward
