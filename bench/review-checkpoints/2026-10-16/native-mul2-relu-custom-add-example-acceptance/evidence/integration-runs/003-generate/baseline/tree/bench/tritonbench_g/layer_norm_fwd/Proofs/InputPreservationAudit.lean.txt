import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.layer_norm_fwd.Proofs.Residual
import bench.tritonbench_g.layer_norm_fwd.Proofs.InputPreservation

/-! Residual execution and explicit input-preservation gates. Full source
composition and the affine output contract remain separate obligations. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd

#axiomsClean InputSuffix.residualStmt_exec
#axiomsClean InputSuffix.residualState_x
#axiomsClean InputSuffix.residualState_mem
#axiomsClean InputSuffix.residualState_regs
#axiomsClean InputSuffix.realStoreStmt_exec
#axiomsClean InputSuffix.residualOutStmt_exec
#axiomsClean InputSuffix.residualOutState_mem
#axiomsClean InputSuffix.residualOutState_regs
#axiomsClean Domain.readsUnchanged_refl
#axiomsClean Domain.readsUnchanged_trans
#axiomsClean Domain.readsUnchanged_of_mem
#axiomsClean Domain.readsUnchanged_readMem
#axiomsClean Domain.readsUnchanged_seed
#axiomsClean Domain.writeCell_readsUnchanged
#axiomsClean Domain.scatter_readsUnchanged
#axiomsClean Execution.loadState_pointers
#axiomsClean Execution.firstStreamState_regs
#axiomsClean Execution.firstStreamState_row
#axiomsClean Execution.firstStreamState_cols
#axiomsClean Execution.firstStreamState_readsUnchanged
#axiomsClean InputSuffix.bodyState_regs
#axiomsClean InputSuffix.bodyState_readsUnchanged
#axiomsClean InputSuffix.streamValues_snapshot
#axiomsClean InputSuffix.rowValues_residual_snapshot

end VeriTile.Bench.TritonBenchG.LayerNormFwd
