import bench.tritonbench_g.pow_scalar_tensor.Proofs.Execution

namespace VeriTile.Bench.TritonBenchG.PowScalarTensor.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 6000
set_option linter.unusedSimpArgs false

/-- Destination storage tags are independent of the numerical power value. -/
def outputCell (output : TypedSource.FloatingOutput) (x : ℝ) : Cell (WithBot ℝ) :=
  match output with
  | .fp32 | .fp64 => .mk .real (some x)
  | .fp16 => .mk .fp16 (some x)
  | .bf16 => .mk .bf16 (some x)
  | .f8e4 => .mk .f8e4 (some x)
  | .f8e5 => .mk .f8e5 (some x)

/-- Numerical full-source executions preserve the destination's actual type.
The multi-tile branch produces 4,8,2; the one-tile branch writes only 4. Masked,
skipped and before-output cells keep integer sentinels, and the input remains. -/
theorem numeric_cells (output : TypedSource.FloatingOutput) (oneTile candidate : Bool) :
    (numeric_run output oneTile candidate).val.mem "Y" 6 = outputCell output 4 ∧
    (numeric_run output oneTile candidate).val.mem "Y" 9 =
      (if oneTile then .mk .int 37 else outputCell output 8) ∧
    (numeric_run output oneTile candidate).val.mem "Y" 18 =
      (if oneTile then .mk .int 37 else outputCell output 2) ∧
    (numeric_run output oneTile candidate).val.mem "Y" 21 = .mk .int 37 ∧
    (numeric_run output oneTile candidate).val.mem "Y" 12 = .mk .int 37 ∧
    (numeric_run output oneTile candidate).val.mem "Y" 0 = .mk .int 37 ∧
    (numeric_run output oneTile candidate).val.mem "X" 4 = .mk .real (some 2) := by
  cases output <;> cases oneTile <;> cases candidate
  all_goals simp (maxSteps := 300000) only [numeric_run, FP.Structural.store,
    State.write, State.setReg_mem, Region.cast, TileShape.allIndices,
    List.finRange_succ, List.foldl_cons, List.foldl_nil, List.flatMap_cons,
    List.flatMap_nil, List.map_cons, List.map_nil, List.cons_append,
    List.nil_append, bop, Broadcast.leftIndex, Broadcast.rightIndex,
    Tile.scalar_data, Tile.select, Tile.ptrAdd, ofFloat, toFloat,
    TileShape.indexToList, BlockPtr.address_1d, BlockPtr.inBounds_1d]
  all_goals simp (maxSteps := 300000) [initial, model, State.setReg,
    outputCell, Region.cast, Function.comp_def, Tile.scalar, -Tile.scalar_eta,
    WithBot.realPow, Real.rpow_two, Real.rpow_natCast, TileShape.indexToList,
    BlockPtr.address_1d, BlockPtr.inBounds_1d]
  all_goals norm_num [Real.rpow_natCast]

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.PowScalarTensor.NumericExecution
