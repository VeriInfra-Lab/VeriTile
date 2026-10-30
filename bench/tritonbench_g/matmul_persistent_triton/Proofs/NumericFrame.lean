import bench.tritonbench_g.matmul_persistent_triton.Proofs.NumericExecution

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 8000000
set_option maxRecDepth 6000
set_option linter.unusedSimpArgs false

/-- The full raw typed frame of the same four-iteration execution: every
input cell, every earlier row, all stride gaps and masked lanes retain their
original tag and value. Only the three active stores may change memory. -/
theorem numeric_frame (storage : TypedSource.Storage) (r : RegionName) (o : Nat)
    (h : r ≠ "C" ∨ (o ≠ 14 ∧ o ≠ 16 ∧ o ≠ 18)) :
    (numeric_run storage).val.mem r o = (initial storage).mem r o := by
  cases storage
  all_goals rcases h with hr | ⟨h14, h16, h18⟩
  all_goals simp (maxSteps := 300000) [numeric_run, prefix_run, iteration0_schedule_run, iteration0_loads_run, iteration0_dot_run, iteration0_store_run, iteration1_schedule_run, iteration1_loads_run, iteration1_dot_run, iteration1_store_run, iteration2_schedule_run, iteration2_loads_run, iteration2_dot_run, iteration2_store_run, iteration3_schedule_run, iteration3_loads_run, iteration3_dot_run, iteration3_store_run,
    FP.Structural.store, State.write, State.setReg_mem,
    Region.cast, TileShape.allIndices, List.finRange_succ,
    List.foldl_cons, List.foldl_nil, List.flatMap_cons, List.flatMap_nil,
    List.map_cons, List.map_nil, List.cons_append, List.nil_append,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Tile.scalar_data,
    integralFloorDiv, integralMod, ofFloat, toFloat, channel, initial, *]

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NumericExecution
