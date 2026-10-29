import bench.repairs.tritonbench_g.chunk_gla_fwd.Misc.NativeCorrect

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.FPEquiv
noncomputable section

/-- Compatibility names retain the exact full producer and consumer IO claims. -/
alias producer_native_observation := NativeCorrect.producer_native_observation
alias consumer_native_observation := NativeCorrect.consumer_native_observation

end
end VeriTile.Bench.Repairs.ChunkGlaFwd.FPEquiv
