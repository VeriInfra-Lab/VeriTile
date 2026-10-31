import bench.tritonbench_g.rope_transform.Kernels

/-! All storage profiles and both rotation directions use the selected source. -/
namespace VeriTile.Bench.TritonBenchG.RopeTransform.Typed

abbrev source := VeriTile.Bench.Repairs.RopeTransform.Kernels.source

theorem source_selected : source=VeriTile.Bench.Repairs.RopeTransform.Kernels.source := rfl

end VeriTile.Bench.TritonBenchG.RopeTransform.Typed
