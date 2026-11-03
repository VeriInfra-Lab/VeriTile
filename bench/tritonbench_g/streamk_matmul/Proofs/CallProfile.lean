import bench.tritonbench_g.streamk_matmul.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.StreamK.TileChecked.CallProfile
open VeriTile.Triton

theorem original_bundle :
    SourceAdapter.decodeSources swizzle_tileSource linear_tileSource mac_loopSource
      first_waveSource full_tilesSource reduce_streamkSource=.ok kernels := kernelsDecoded.property

theorem candidate_bundle :
    SourceAdapter.decodeSources area_swizzle_tileSource area_linear_tileSource area_mac_loopSource
      area_first_waveSource area_full_tilesSource area_reduce_streamkSource=.ok areaKernels := areaKernelsDecoded.property

theorem missing_dtype_rejected :
    SourceAdapter.decode {mac_loopSource with dtypeConstants:=[]}=
      .error "unsupported StreamK caller ABI, dtype or helper binding" := rfl

theorem wrong_dtype_rejected :
    SourceAdapter.decode {mac_loopSource with dtypeConstants:=[("ACC_TYPE",.fp16)]}=
      .error "unsupported StreamK caller ABI, dtype or helper binding" := rfl

theorem wrong_pointer_rejected :
    SourceAdapter.decode {mac_loopSource with signature:={mac_loopSource.signature with
      pointers:=[("A",.fp16),("B",.fp32),("C",.fp32),("partials",.fp32)]}}=
      .error "unsupported StreamK caller ABI, dtype or helper binding" := rfl

theorem unlinked_helper_rejected :
    SourceAdapter.decodeSources swizzle_tileSource linear_tileSource mac_loopSource
      {first_waveSource with helperBindings:=[("mac_loop_area","mac")]}
      full_tilesSource reduce_streamkSource=
      .error "StreamK helper does not link to the supplied source member" := rfl

#axiomsClean original_bundle
#axiomsClean candidate_bundle
#axiomsClean missing_dtype_rejected
#axiomsClean wrong_dtype_rejected
#axiomsClean wrong_pointer_rejected
#axiomsClean unlinked_helper_rejected
end VeriTile.Bench.Repairs.StreamK.TileChecked.CallProfile
