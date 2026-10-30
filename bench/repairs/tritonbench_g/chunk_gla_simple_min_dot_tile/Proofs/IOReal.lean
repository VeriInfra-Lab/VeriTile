import bench.repairs.tritonbench_g.chunk_gla_simple_min_dot_tile.Proofs.IOBindings
import bench.repairs.tritonbench_g.chunk_gla_simple_min_dot_tile.Proofs.IORealGrid
import bench.repairs.tritonbench_g.chunk_gla_simple_min_dot_tile.Proofs.IOSourceSafety

namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.PublicIO
open VeriTile Triton Host IOBindings
open VeriTile.Bench.TritonBenchG.ChunkGlaSimple Math Real
noncomputable section
set_option maxHeartbeats 1600000
set_option maxRecDepth 24000

theorem real_implements (c : Shape) (bt : 0<c.BT) (scale : ℝ) :
    (realIO c scale).RegionalScheduledImplements (fun _ => realSchedule c)
      (PublicMath.formula c bt scale) := by
  apply UKernelIO.RegionalScheduledImplements.intro
  · intro s xs _ bounds hb
    change AlgorithmScheduleSafe (DirectReal.kernel c scale) bounds (realSchedule c) s
    apply algorithmScheduleSafe_of_forall
    intro job member current
    obtain ⟨q, _, rfl⟩ := List.mem_map.mp member
    change (DirectReal.kernel c scale).TraceSafeR .triv bounds (current.withGridIndex q)
    rw [DirectReal.kernel, DirectReal.source_binding]
    apply IOSourceSafety.source_safe (layout c) regions
    rw [launch_grid]
    exact capacity_bounds c q bounds (real_capacities c scale xs s bounds hb)
  · intro s xs hin
    obtain ⟨t, he, hv, hf⟩ := IORealGrid.complete c bt scale s
    refine ⟨t, he, ?_, ?_⟩
    · intro o i _
      change t.readMem "o" i.val = PublicMath.formula c bt scale xs o i
      rw [← output_index c i]
      exact (hv _ _ _).trans (real_formula c bt scale xs s hin _ _ _)
    · intro r off out _
      apply (hf r off ?_).symm
      rintro ⟨hr,hi⟩
      change r="o" at hr
      subst r
      have h := out (0 : Fin 1) ⟨off,hi⟩ trivial
      exact h.elim (fun h => h rfl) (fun h => h rfl)

#axiomsClean real_implements
end
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.PublicIO
