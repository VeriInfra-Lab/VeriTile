import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.SourceCorrect
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.HistoricalFPEquiv

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention
-- Unified entry: full mathematical API, returned VJP, FP32 words and frames.
#axiomsClean sourceCorrect
#axiomsClean returned_vjp
#axiomsClean fp32_public_words
#axiomsClean fp32_backend_equiv
#specNonCircular mathematicalResult avoiding [forwardAPI,backwardAPI,runtimeForward,runtimeBackward]
end VeriTile.Bench.Repairs.ParallelRetentionAttention
