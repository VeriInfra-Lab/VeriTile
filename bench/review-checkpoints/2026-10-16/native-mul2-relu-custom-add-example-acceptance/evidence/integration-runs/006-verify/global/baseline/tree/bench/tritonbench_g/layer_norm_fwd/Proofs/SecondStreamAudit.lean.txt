import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.layer_norm_fwd.Proofs.SecondStream

/-! The source subtree and second-stream phase gates, without a claim that
the preceding memory frame or the remaining full-kernel contract is complete. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd

#axiomsClean InputSuffix.source_phase
#axiomsClean InputSuffix.rowLoadOp_eval
#axiomsClean InputSuffix.secondRandomOp_eval
#axiomsClean InputSuffix.secondKeepOp_eval
#axiomsClean InputSuffix.dropoutValueOp_eval
#axiomsClean InputSuffix.maskStmt_exec
#axiomsClean InputSuffix.dropoutPhase_exec
#axiomsClean InputSuffix.dropoutState_x1
#axiomsClean InputSuffix.scalePhase_exec
#axiomsClean InputSuffix.scaledState_x1
#axiomsClean InputSuffix.scaledState_mem
#axiomsClean InputSuffix.scaledState_seed
#axiomsClean InputSuffix.droppedState_x1
#axiomsClean InputSuffix.droppedState_regs
#axiomsClean InputSuffix.addStmt_exec
#axiomsClean InputSuffix.secondBody_exec
#axiomsClean InputSuffix.bodyState_x
#axiomsClean InputSuffix.bodyState_mem

end VeriTile.Bench.TritonBenchG.LayerNormFwd
