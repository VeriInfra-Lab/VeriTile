import bench.repairs.tritonbench_g.rotary_transform_ops.Proofs.Scheduling

namespace VeriTile.Bench.Repairs.RotaryTransformOps

/-- A permutation of complete source programs preserves the independent
rotation result, including inplace execution. Each program retains its
load-before-store order. -/
theorem sourcePrograms_order {α : Type} (c : Config) (valid : c.Valid) (p : Ops α)
    (order : List Program) (complete : (programs c).Perm order) (m : Memory α) :
    sourcePrograms c p order m = source c p m := by
  rw [sourcePrograms_eq c valid p order (fun _ h => complete.mem_iff.mpr h)]
  apply any_grid_order c valid p _ _ m
  rw [schedule_flat]
  exact complete.flatMap_right (programWork c)

#axiomsClean sourcePrograms_order
end VeriTile.Bench.Repairs.RotaryTransformOps
