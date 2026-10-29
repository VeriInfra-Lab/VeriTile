import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.layer_norm_fwd.Proofs.NormalizationExecution

/-! Phase gates only: these do not discharge incoming combined-value or
remaining affine-output, finite-IO, or typed execution obligations. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd

#axiomsClean Normalization.source_phase
#axiomsClean Normalization.centerVariance_exec
#axiomsClean Normalization.mean_exec
#axiomsClean Normalization.statistics_exec
#axiomsClean Normalization.statisticsState_var
#axiomsClean Normalization.statisticsState_row
#axiomsClean Normalization.phase_exec
#axiomsClean Normalization.source_phase_exec
#axiomsClean Normalization.state_mem
#axiomsClean Normalization.state_rstd
#axiomsClean Normalization.state_xbar
#axiomsClean Normalization.state_x
#axiomsClean Normalization.state_mean

end VeriTile.Bench.TritonBenchG.LayerNormFwd
