import bench.tritonbench_g.pow_scalar_tensor.Proofs.IntegerExecution
import bench.tritonbench_g.pow_scalar_tensor.Proofs.ExecutionValues

namespace VeriTile.Bench.TritonBenchG.PowScalarTensor.IntegerExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 6000
set_option linter.unusedSimpArgs false

abbrev outputCell := NumericExecution.outputCell

/-- Numerical full-source executions preserve the destination's actual type.
The multi-tile branch produces 1/4,1/8,1/2; the one-tile branch writes only 1/4. Masked,
skipped and before-output cells keep integer sentinels, and the input remains. -/
theorem numeric_cells (output : TypedSource.FloatingOutput) (oneTile candidate : Bool) :
    (numeric_run output oneTile candidate).val.mem "Y" 6 = outputCell output (1/4) ∧
    (numeric_run output oneTile candidate).val.mem "Y" 9 =
      (if oneTile then .mk .int 37 else outputCell output (1/8)) ∧
    (numeric_run output oneTile candidate).val.mem "Y" 18 =
      (if oneTile then .mk .int 37 else outputCell output (1/2)) ∧
    (numeric_run output oneTile candidate).val.mem "Y" 21 = .mk .int 37 ∧
    (numeric_run output oneTile candidate).val.mem "Y" 12 = .mk .int 37 ∧
    (numeric_run output oneTile candidate).val.mem "Y" 0 = .mk .int 37 ∧
    (numeric_run output oneTile candidate).val.mem "X" 4 = .mk .int (-2) := by
  cases output <;> cases oneTile <;> cases candidate
  all_goals simp (maxSteps := 300000) only [numeric_run, FP.Structural.store,
    State.write, State.setReg_mem, Region.cast, TileShape.allIndices,
    List.finRange_succ, List.foldl_cons, List.foldl_nil, List.flatMap_cons,
    List.flatMap_nil, List.map_cons, List.map_nil, List.cons_append,
    List.nil_append, bop, Broadcast.leftIndex, Broadcast.rightIndex,
    Tile.scalar_data, Tile.select, Tile.ptrAdd, ofFloat, toFloat,
    TileShape.indexToList, BlockPtr.address_1d, BlockPtr.inBounds_1d]
  all_goals simp (maxSteps := 300000) [initial, model, NumericExecution.model, State.setReg,
    outputCell, NumericExecution.outputCell, Region.cast, Function.comp_def, Tile.scalar, -Tile.scalar_eta,
    WithBot.realPowi, WithBot.map, TileShape.indexToList,
    BlockPtr.address_1d, BlockPtr.inBounds_1d]
  all_goals norm_num

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.PowScalarTensor.IntegerExecution
