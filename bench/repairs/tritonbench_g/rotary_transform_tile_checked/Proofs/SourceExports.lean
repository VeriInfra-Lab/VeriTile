import bench.repairs.tritonbench_g.rotary_transform_tile_checked.Proofs.KernelIndex

/-! Compatibility proof exports formerly mixed into the source-only facade. -/
namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked
noncomputable section
abbrev rotary_transform_source := TypedExecution.kernel
abbrev rotary_transform_grid := @TypedGrid.run
abbrev rotary_transform_grid_order := @TypedGrid.complete_order
#axiomsClean rotary_transform_grid_order
end
end VeriTile.Bench.Repairs.RotaryTransformTileChecked
