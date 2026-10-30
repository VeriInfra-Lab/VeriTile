import bench.tritonbench_g.nested_loops_processing.Proofs.SourceModel
import bench.tritonbench_g.nested_loops_processing.Proofs.Mathematical_Math

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing
open VeriTile.Bench.TritonBenchG.NestedLoopsProcessing
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000 in
theorem source_store_schedule (input : Nat → Tile α) (zero : α) :
    sourceStores input zero=(List.range 24).map (fun t=>(2*t,input (CopyMath.inputShift t))) := by
  simp [sourceStores,program,outer,middle,inner,traceStep,initialRegisters,
    Function.update,CopyMath.inputShift,List.range_succ,List.foldl_append]

theorem source_final_offsets (input : Nat → Tile α) (zero : α) :
    (program.foldl (traceStep input) (initialRegisters zero,[])).1.inputShift=28 ∧
    (program.foldl (traceStep input) (initialRegisters zero,[])).1.outputShift=48 := by
  exact ⟨rfl,rfl⟩

theorem source_load_store_count :
    (program.filter (fun i=>match i with | .load _=>true | _=>false)).length=14 ∧
    (program.filter (fun i=>match i with | .store _=>true | _=>false)).length=24 := by
  decide

theorem source_barrier_count :
    (program.filter (fun i=>match i with | .ctaBarrier=>true | _=>false)).length=24 := by
  decide

/-- Each source store is immediately followed by uniform CTA synchronization. -/
theorem source_store_barrier_pairs :
    (program.filter (fun i=>match i with | .store _ | .ctaBarrier=>true | _=>false))=
      List.flatten (List.replicate 8 [.store .a1,.ctaBarrier,.store .a2,.ctaBarrier,.store .a3,.ctaBarrier]) := by
  decide

#axiomsClean source_barrier_count
#axiomsClean source_store_barrier_pairs
#axiomsClean source_store_schedule
#axiomsClean source_final_offsets
#axiomsClean source_load_store_count
end
end VeriTile.Bench.Repairs.NestedLoopsProcessing
