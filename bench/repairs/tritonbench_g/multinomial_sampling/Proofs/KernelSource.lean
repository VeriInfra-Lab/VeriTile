import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPSource

namespace VeriTile.Bench.Repairs.MultinomialSampling
open VeriTile.Triton SourceValue

/-- The complete selected FP32 kernel, with its source parameters. -/
def kernel (p : Parameters) : ComputeKernel := source .fp32 p

theorem kernel_source (p : Parameters) : kernel p =
    Source.original p.scores p.seeds p.offsets p.indices p.output
      p.strideScoreBatch p.strideScoreToken p.strideIndexBatch p.strideIndexToken
      p.batches p.tokens p.block p.blockTokens p.blockDomain p.tokensDomain := rfl

#axiomsClean kernel_source
end VeriTile.Bench.Repairs.MultinomialSampling
