import bench.tritonbench_g.streamk_matmul.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.StreamK.TileChecked
set_option maxRecDepth 50000
/-- Every complete helper and stage source lowers to its actual execution program. -/
theorem kernels_source_preserved :
    SourceAdapter.decode swizzle_tileSource = .ok kernels.swizzle ∧
    SourceAdapter.decode linear_tileSource = .ok kernels.linear ∧
    SourceAdapter.decode mac_loopSource = .ok kernels.mac ∧
    SourceAdapter.decode first_waveSource = .ok kernels.first ∧
    SourceAdapter.decode full_tilesSource = .ok kernels.«full» ∧
    SourceAdapter.decode reduce_streamkSource = .ok kernels.reduce :=
  ⟨swizzle_tileDecoded.property, linear_tileDecoded.property, mac_loopDecoded.property, first_waveDecoded.property, full_tilesDecoded.property, reduce_streamkDecoded.property⟩
theorem area_kernels_source_preserved :
    SourceAdapter.decode swizzle_tileSource = .ok areaKernels.swizzle ∧
    SourceAdapter.decode linear_tileSource = .ok areaKernels.linear ∧
    SourceAdapter.decode area_mac_loopSource = .ok areaKernels.mac ∧
    SourceAdapter.decode first_waveSource = .ok areaKernels.first ∧
    SourceAdapter.decode full_tilesSource = .ok areaKernels.«full» ∧
    SourceAdapter.decode area_reduce_streamkSource = .ok areaKernels.reduce :=
  ⟨swizzle_tileDecoded.property, linear_tileDecoded.property, area_mac_loopDecoded.property, first_waveDecoded.property, full_tilesDecoded.property, area_reduce_streamkDecoded.property⟩

#axiomsClean kernels_source_preserved
#axiomsClean area_kernels_source_preserved
end VeriTile.Bench.Repairs.StreamK.TileChecked
