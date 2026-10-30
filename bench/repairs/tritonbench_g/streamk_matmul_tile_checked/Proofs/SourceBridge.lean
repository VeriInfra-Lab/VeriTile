import bench.repairs.tritonbench_g.streamk_matmul_tile_checked.Proofs.Reducer

namespace VeriTile.Bench.Repairs.StreamK.TileChecked.Reader
open Whole001 Source VeriTile.Bench.TritonBenchG.StreamK
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 20000

theorem selected_parts (p : Source.Parameters) :
    (Plan.select p.h).fullIterations=(Plan.select p.h).streamTiles*(Plan.select p.h).iterations/p.h.programs ∧
    (Plan.select p.h).partialIterations=(Plan.select p.h).streamTiles*(Plan.select p.h).iterations%p.h.programs := by
  by_cases hz : p.h.k=0 <;> simp [Plan.select,hz,hostPlan]

theorem program_events (p : Source.Parameters) (ops : Instructions α) (program : Fin p.h.programs)
    (hI : 0<(Plan.select p.h).iterations) :
    (Protocol.chunks (Plan.select p.h).iterations
      (Plan.edge (Plan.select p.h).fullIterations (Plan.select p.h).partialIterations (program.val+1))
      (Plan.edge (Plan.select p.h).fullIterations (Plan.select p.h).partialIterations program.val) hI).flatMap
      (macEvents p sourcePlan.mac ops program) =
    (Protocol.programChunks (Plan.select p.h).streamTiles (Plan.select p.h).iterations p.h.programs program.val hI).flatMap
      (macLoop p ops program) := by
  rw [(selected_parts p).1,(selected_parts p).2]
  apply List.flatMap_congr
  intro c hc
  have shape := Protocol.member_shape _ _ _ hI c hc
  have bound : c.hi≤Protocol.boundary (Plan.select p.h).iterations c.lo := by
    rw [shape.2.2.2,Protocol.next_eq _ _ _ hI]
    exact min_le_left _ _
  exact mac_events p program c ops hI bound

theorem first_events (p : Source.Parameters) (ops : Instructions α) :
    firstEvents p sourcePlan ops = some (firstWave p ops) := by
  by_cases active : 0<(Plan.select p.h).streamTiles
  · have hI := Plan.iterations_positive p.h (Plan.selected_active p.h active).2 p.blockK
    unfold firstEvents
    rw [if_pos active]
    simp only [first_pid,first_stop,first_start,source_chunks p _ _ _ hI,Option.bind_eq_bind,Option.bind_some,Option.pure_def,program_events]
    have mapPure (f : Fin p.h.programs → List (Source.Event α p)) :
        (List.finRange p.h.programs).mapM (fun program => some (f program))=
          some ((List.finRange p.h.programs).map f) :=
      List.mapM_pure (m:=Option) (f:=f)
    rw [mapPure,Option.bind_some]
    unfold firstWave
    rw [dif_pos active]
    rfl
  · simp only [firstEvents,if_neg active,firstWave,dif_neg active]

theorem all_events (p : Source.Parameters) (ops : Instructions α) :
    events p sourcePlan ops=some ⟨firstWave p ops,reduceWave p,fullWave p ops⟩ := by
  simp only [events,source_typed,if_true,first_events,Option.bind_eq_bind,Option.bind_some,Option.pure_def]
  congr 2
  · simp only [reduceWave,reducePairs,List.map_flatMap]
    apply List.flatMap_congr
    intro tile ht
    exact reduce_events p tile
  · unfold fullWave
    apply List.flatMap_congr
    intro pid hp
    exact full_events p pid ops

/-- Every source statement is decoded and interpreted before reaching the
previously proved controller. Source success and this whole projection are
conclusions, never admission hypotheses. -/
theorem execute_projection (p : Source.Parameters) (ops : Instructions α)
    (initial : Source.State α p exterior) :
    execute kernels p ops initial=some (Source.call p ops initial) := by
  unfold execute
  rw [source_compiles]
  simp only [Option.bind_eq_bind,Option.bind_some,Option.pure_def]
  unfold Source.call
  split
  · rfl
  · rw [all_events]
    rfl

#axiomsClean program_events
#axiomsClean first_events
#axiomsClean all_events
#axiomsClean execute_projection
#specNonCircular execute avoiding [Source.call,Source.firstWave,Source.fullWave,Source.reduceWave]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.Reader
