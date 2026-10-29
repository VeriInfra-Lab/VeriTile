import bench.repairs.tritonbench_g.attention_score.Kernels
import bench.repairs.tritonbench_g.attention_score.Proofs.RealSource
namespace VeriTile.Bench.Repairs.AttentionScore.Whole002.TypedSource
open VeriTile.Triton
/-- The DSL uses mathematical index integers. Widening changes machine
representation, whose bounds are proved separately, but not this source tree. -/
theorem widened_eq : widened = AttentionScore.TypedSource.original := by
  funext Q K M Out sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N R NK B D BN off sz scale w comp em en eq
  simp only [widened, AttentionScore.TypedSource.original]
#axiomsClean widened_eq
end VeriTile.Bench.Repairs.AttentionScore.Whole002.TypedSource
