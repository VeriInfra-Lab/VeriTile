import bench.repairs.tritonbench_g.chunk_delta_fwd.Kernels

/-! Both complete raw Python bodies are in the shared ChunkDeltaFwd.Kernels
source module. The selected wrapper and existing constexpr-address candidate
reuse these definitions without reconstructing their statement lists. -/
namespace VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot
open VeriTile Triton
abbrev Parameters := ChunkDeltaFwd.Parameters
abbrev kernel := ChunkDeltaFwd.kernel
abbrev addressFold := ChunkDeltaFwd.addressFold

theorem shared_source (p : Parameters) : kernel p = ChunkDeltaFwd.kernel p := rfl
#axiomsClean shared_source
end VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot
