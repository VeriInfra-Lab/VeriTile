import bench.tritonbench_g.matmul_persistent_triton.Proofs.NumericExecution

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 8000000
set_option maxRecDepth 6000
set_option linter.unusedSimpArgs false

private theorem sum_fin2 {α : Type*} [AddCommMonoid α]
    (I : Fintype (Fin 2)) (f : Fin 2 → α) :
    @Finset.sum (Fin 2) α _ (@Finset.univ _ I) f = f 0 + f 1 := by
  have hI : I = Fin.fintype 2 := Subsingleton.elim _ _
  rw [hI]
  simp [Fin.sum_univ_succ]

/-- Actual nonconstant matrix-product readbacks: row 2 has outputs 18,27,36.
The third output is in a later persistent tile after an accumulator reset. -/
theorem numeric_cells (storage : TypedSource.Storage) :
    (numeric_run storage).val.mem "C" 14 =
      .mk (channel storage).toTileDType (ofFloat (channel storage) (some 18)) ∧
    (numeric_run storage).val.mem "C" 16 =
      .mk (channel storage).toTileDType (ofFloat (channel storage) (some 27)) ∧
    (numeric_run storage).val.mem "C" 18 =
      .mk (channel storage).toTileDType (ofFloat (channel storage) (some 36)) ∧
    (numeric_run storage).val.mem "C" 20 = .mk .int 37 ∧
    (numeric_run storage).val.mem "C" 15 = .mk .int 37 := by
  cases storage
  all_goals simp (maxSteps := 300000) only [numeric_run, prefix_run, iteration0_schedule_run, iteration0_loads_run, iteration0_dot_run, iteration0_store_run, iteration1_schedule_run, iteration1_loads_run, iteration1_dot_run, iteration1_store_run, iteration2_schedule_run, iteration2_loads_run, iteration2_dot_run, iteration2_store_run, iteration3_schedule_run, iteration3_loads_run, iteration3_dot_run, iteration3_store_run,
    FP.Structural.store, State.write, State.setReg_mem,
    Region.cast, TileShape.allIndices, List.finRange_succ,
    List.foldl_cons, List.foldl_nil, List.flatMap_cons, List.flatMap_nil,
    List.map_cons, List.map_nil, List.cons_append, List.nil_append,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Tile.scalar_data,
    ofFloat, toFloat, channel]
  all_goals simp (maxSteps := 300000) [initial, model, State.setReg, Region.cast,
    Function.comp_def, Tile.scalar, -Tile.scalar_eta, sum_fin2, Tile.dot,
    integralFloorDiv, integralMod, channel, ofFloat, toFloat,
    WithBot.realAdd, WithBot.realSub, WithBot.realMul, WithBot.realDiv]
  all_goals norm_num [WithBot.some_eq_coe, ← WithBot.coe_add, Option.map₂, Option.map,
    Option.bind, FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    Tile.scalar, -Tile.scalar_eta, Function.comp_def, List.finRange_succ,
    List.foldl, List.flatMap, sum_fin2, integralFloorDiv, integralMod, channel, ofFloat, toFloat]

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NumericExecution
