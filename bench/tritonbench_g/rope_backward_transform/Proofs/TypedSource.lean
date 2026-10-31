import bench.tritonbench_g.rope_backward_transform.Kernels

/-! All storage profiles and both directions use the shared repaired source. -/
namespace VeriTile.Bench.Repairs.RopeBackwardTransform.Source

abbrev source := VeriTile.Bench.Repairs.RopeBackwardTransform.Kernels.source

theorem source_selected : source=VeriTile.Bench.Repairs.RopeBackwardTransform.Kernels.source := rfl

end VeriTile.Bench.Repairs.RopeBackwardTransform.Source
