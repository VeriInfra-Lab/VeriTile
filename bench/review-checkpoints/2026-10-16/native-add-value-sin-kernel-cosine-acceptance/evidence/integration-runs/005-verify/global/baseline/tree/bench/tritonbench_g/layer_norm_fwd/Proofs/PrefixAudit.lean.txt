import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.layer_norm_fwd.Proofs.NormPrefix

/-! Composition gates for the actual first eighteen source statements.
These establish the normalization input by executing the complete input
prefix; the remaining affine outputs are still a separate proof obligation. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd

#axiomsClean InputPrefix.secondStmt_exec
#axiomsClean InputPrefix.secondState_regs
#axiomsClean InputPrefix.secondState_readsUnchanged
#axiomsClean InputPrefix.secondState_x
#axiomsClean InputPrefix.afterResidual_regs
#axiomsClean InputPrefix.afterResidual_readsUnchanged
#axiomsClean InputPrefix.afterResidual_x
#axiomsClean InputPrefix.state_regs
#axiomsClean InputPrefix.state_x
#axiomsClean InputPrefix.phase_exec
#axiomsClean InputPrefix.source_exec
#axiomsClean InputPrefix.secondState_mem
#axiomsClean InputPrefix.state_readsUnchanged
#axiomsClean InputPrefix.state_mem
#axiomsClean NormPrefix.source_exec
#axiomsClean NormPrefix.state_regs
#axiomsClean NormPrefix.state_x
#axiomsClean NormPrefix.state_mean
#axiomsClean NormPrefix.state_rstd
#axiomsClean NormPrefix.state_mem
#axiomsClean NormPrefix.state_readsUnchanged

end VeriTile.Bench.TritonBenchG.LayerNormFwd
