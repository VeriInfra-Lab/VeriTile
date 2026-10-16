import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.layer_norm_fwd.Proofs.RealSurface

namespace VeriTile.Bench.TritonBenchG.LayerNormFwd

#axiomsClean RealSurface.erase_projection
#axiomsClean RealSurface.realizes_of_contract
#axiomsClean RealSurface.original_realizes
#axiomsClean RealSurface.candidate_realizes

end VeriTile.Bench.TritonBenchG.LayerNormFwd
