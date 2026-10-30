import bench.repairs.tritonbench_g.bgmv_expand_slice.Kernels
namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked
open VeriTile.Triton
abbrev Call := VeriTile.Bench.Repairs.BgmvExpandSlice.Call
namespace Kernels
abbrev original := VeriTile.Bench.Repairs.BgmvExpandSlice.Kernels.original
abbrev hostCast := VeriTile.Bench.Repairs.BgmvExpandSlice.Kernels.hostCast
abbrev Primary := VeriTile.Bench.Repairs.BgmvExpandSlice.Kernels.Primary
abbrev split_length := VeriTile.Bench.Repairs.BgmvExpandSlice.Kernels.split_length
end Kernels
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked
