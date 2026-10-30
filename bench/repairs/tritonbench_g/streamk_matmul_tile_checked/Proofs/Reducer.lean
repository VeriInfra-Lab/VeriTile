import bench.repairs.tritonbench_g.streamk_matmul_tile_checked.Proofs.Stores

namespace VeriTile.Bench.Repairs.StreamK.TileChecked.Reader
open Whole001 Source
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 20000
local instance (q r I tile program : Nat) : Decidable (Protocol.readsScratch q r I tile program) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _))

theorem reduce_condition (p : Source.Parameters) (tile program : Nat) (lane : Lane p) :
    sourcePlan.reduce.condition.eval (reduceEnv p sourcePlan.reduce tile program lane) (mlane tile lane 0) ↔
      Protocol.readsScratch (Plan.select p.h).fullIterations (Plan.select p.h).partialIterations
        (Plan.select p.h).iterations tile program := by
  change (Plan.edge (Plan.select p.h).fullIterations (Plan.select p.h).partialIterations (0+program) <
      Plan.edge (Plan.select p.h).fullIterations (Plan.select p.h).partialIterations (0+program+1) ∧
    Plan.edge (Plan.select p.h).fullIterations (Plan.select p.h).partialIterations (0+program+1)%(Plan.select p.h).iterations≠0) ∧
    Plan.edge (Plan.select p.h).fullIterations (Plan.select p.h).partialIterations (0+program+1)/(Plan.select p.h).iterations=tile ↔ _
  simp only [Nat.zero_add,Protocol.readsScratch,and_assoc]

theorem reduce_mask (p : Source.Parameters) (tile program : Nat) (lane : Lane p) :
    sourcePlan.reduce.atomicMask.eval (reduceEnv p sourcePlan.reduce tile program lane) (mlane tile lane 0) ↔
      active p tile lane := by
  change (if active p tile lane then 1 else 0)≠0 ↔ _
  by_cases h : active p tile lane <;> simp [h]

theorem reduce_scratch_index (p : Source.Parameters) (tile : Nat) (program : Fin p.h.programs) (lane : Lane p) :
    sourcePlan.reduce.load.index (reduceEnv p sourcePlan.reduce tile program.val lane)
      zeroPointers (mlane tile lane 0)=(scratch p program lane).val := by
  change 0+((0+program.val)*p.h.bm*p.h.bn+lane.1.val*p.h.bn+lane.2.val)=_
  simp only [Nat.zero_add,scratch]

theorem reduce_output_index (p : Source.Parameters) (tile program : Nat) (lane : Lane p)
    (ha : active p tile lane) :
    sourcePlan.reduce.output.index (reduceEnv p sourcePlan.reduce tile program lane)
      zeroPointers (mlane tile lane 0)=(output p tile lane ha).val := by
  change 0+(row p tile lane*max p.h.n 1+col p tile lane*1)=row p tile lane*p.h.n+col p tile lane
  have hn : max p.h.n 1=p.h.n := max_eq_left (by have := ha.2; omega)
  rw [hn]
  omega

theorem reduce_events (p : Source.Parameters) (tile : Nat) :
    reduceEvents p sourcePlan.reduce tile =
      (tilePairs p tile).map (fun pair => (Memory.Event.atomic pair.1 pair.2 : Source.Event α p)) := by
  unfold reduceEvents tilePairs
  change (List.range' 0 (p.h.programs-0)).flatMap _=_
  simp only [Nat.sub_zero,List.range'_eq_map_range,Nat.zero_add,List.map_id_fun]
  rw [←List.map_coe_finRange_eq_range]
  simp only [List.flatMap_map,List.map_flatMap]
  apply List.flatMap_congr
  intro program hp
  dsimp only [Function.comp_def]
  unfold programPairs
  by_cases hr : Protocol.readsScratch (Plan.select p.h).fullIterations (Plan.select p.h).partialIterations
      (Plan.select p.h).iterations tile program.val
  · rw [if_pos hr,List.map_flatMap]
    apply List.flatMap_congr
    intro lane hl
    simp only [reduce_condition,hr,true_and,reduce_mask]
    split
    next ha =>
      rw [reduce_scratch_index,reduce_output_index p tile program.val lane ha]
      simp only [atomicEvent,dif_pos (output p tile lane ha).isLt,dif_pos (scratch p program lane).isLt,
        List.map_cons,List.map_nil]
    next => rfl
  · simp only [if_neg hr,List.map_nil]
    apply List.flatMap_eq_nil_iff.mpr
    intro lane hl
    simp only [reduce_condition,hr,false_and,if_false]

/-- The source's scratch load is unconditional across all tile lanes. This
safety/result theorem includes lanes whose following output atomic is masked. -/
theorem all_scratch_loads (p : Source.Parameters) (ops : Instructions α)
    (initial : Source.State α p exterior) (tile : Nat)
    (ht : tile<(Plan.select p.h).streamTiles) (program : Fin p.h.programs) (lane : Lane p)
    (condition : sourcePlan.reduce.condition.eval
      (reduceEnv p sourcePlan.reduce tile program.val lane) (mlane tile lane 0)) :
    let offset := sourcePlan.reduce.load.index (reduceEnv p sourcePlan.reduce tile program.val lane)
      zeroPointers (mlane tile lane 0)
    offset<p.h.programs*p.h.bm*p.h.bn ∧
      (Memory.run ops.atomicAdd (firstWave p ops) initial).scratch
        (scratch p program lane)=
      let c := (Reference.partialChunk p tile ht program ((reduce_condition p tile program.val lane).mp condition)).chunk
      chunkValue p ops initial.inputA initial.inputB tile c.lo c.hi lane := by
  dsimp only
  rw [reduce_scratch_index]
  exact ⟨(scratch p program lane).isLt,
    reducer_load_value p ops initial tile ht program ((reduce_condition p tile program.val lane).mp condition) lane⟩

#axiomsClean reduce_events
#axiomsClean all_scratch_loads
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.Reader
