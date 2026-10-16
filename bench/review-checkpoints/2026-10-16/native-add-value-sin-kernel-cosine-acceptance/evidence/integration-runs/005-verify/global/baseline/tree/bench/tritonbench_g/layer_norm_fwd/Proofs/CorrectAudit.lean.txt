import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.layer_norm_fwd.Correct

namespace VeriTile.Bench.TritonBenchG.LayerNormFwd

#axiomsClean Correctness.layer_norm_fwd_original_real
#axiomsClean Correctness.layer_norm_fwd_candidate_real

end VeriTile.Bench.TritonBenchG.LayerNormFwd
