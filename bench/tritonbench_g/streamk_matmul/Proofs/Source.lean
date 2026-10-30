import bench.tritonbench_g.streamk_matmul.Sources.SourceDefinitions
import bench.tritonbench_g.streamk_matmul.Proofs.HostPlan
import bench.tritonbench_g.streamk_matmul.Proofs.Memory
namespace VeriTile.Bench.Repairs.StreamK.Whole001
open VeriTile.Bench.TritonBenchG.StreamK
namespace Source
open Protocol
noncomputable section
theorem call_frame (p : Parameters) (ops : Instructions α) (initial : State α p exterior) :
    (call p ops initial).inputA=initial.inputA ∧ (call p ops initial).inputB=initial.inputB ∧
    (call p ops initial).exterior=initial.exterior := by
  unfold call
  split
  · exact ⟨rfl,rfl,rfl⟩
  · rw [←Memory.ordered_launches]
    exact Memory.run_inputs _ _ _

theorem source_reads_initialized_scratch (p : Parameters)
    (tile program : Nat) (hs : 0 < (Plan.select p.h).streamTiles)
    (hr : readsScratch (Plan.select p.h).fullIterations (Plan.select p.h).partialIterations
      (Plan.select p.h).iterations tile program) :
    ∃! c, c ∈ programChunks (Plan.select p.h).streamTiles (Plan.select p.h).iterations p.h.programs program
      (Plan.iterations_positive p.h (Plan.selected_active p.h hs).2 p.blockK) ∧
      c.hi%(Plan.select p.h).iterations≠0 ∧ c.tile (Plan.select p.h).iterations=tile := by
  have hI:=Plan.iterations_positive p.h (Plan.selected_active p.h hs).2 p.blockK
  have hq:=(Plan.selected_divmod p.h hs).1
  have hr':=(Plan.selected_divmod p.h hs).2
  rw [hq,hr'] at hr
  exact read_has_unique_writer _ _ _ _ _ hI hr

#axiomsClean call_frame
#axiomsClean source_reads_initialized_scratch
#specNonCircular firstWave avoiding [call,Memory.run,Memory.step]
#specNonCircular fullWave avoiding [call,Memory.run,Memory.step]
end
end Source
end VeriTile.Bench.Repairs.StreamK.Whole001
