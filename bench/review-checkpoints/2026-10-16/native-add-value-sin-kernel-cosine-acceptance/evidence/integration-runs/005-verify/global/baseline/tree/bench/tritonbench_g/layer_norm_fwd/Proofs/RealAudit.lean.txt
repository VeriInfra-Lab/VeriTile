import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.layer_norm_fwd.Proofs.Real

/-! Full-source real execution and seven-output initial-snapshot contract,
plus the complete affine phase gates. Finite flat IO, the candidate's separate
real proof and the typed execution witness remain separate obligations. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd

#axiomsClean Affine.source_phase
#axiomsClean Affine.maskStmt_exec
#axiomsClean Affine.coefficientOp_eval
#axiomsClean Affine.storeStmt_exec
#axiomsClean Affine.eval_ite_bool
#axiomsClean Affine.eval_ite_not
#axiomsClean Affine.normalizedOp_eval
#axiomsClean Affine.valueOp_eval
#axiomsClean Affine.loadPhase_exec
#axiomsClean Affine.loadState_weight
#axiomsClean Affine.loadState_bias
#axiomsClean Affine.loadState_regs
#axiomsClean Affine.loadState_mem
#axiomsClean Affine.outputPhase_exec
#axiomsClean Affine.outputState_mem
#axiomsClean Affine.outputState_readsUnchanged
#axiomsClean Affine.firstPhase_exec
#axiomsClean Affine.firstState_regs
#axiomsClean Affine.firstState_xhat
#axiomsClean Affine.firstState_mask
#axiomsClean Affine.firstState_mem
#axiomsClean Affine.firstState_readsUnchanged
#axiomsClean Affine.secondBody_exec
#axiomsClean Affine.secondStmt_exec
#axiomsClean Affine.secondBodyState_mem
#axiomsClean Affine.secondState_mem
#axiomsClean Affine.phase_exec
#axiomsClean Affine.mathematicalXhat_active
#axiomsClean Affine.affineValues_y
#axiomsClean Affine.affineValues_y1
#axiomsClean RealExecution.source_exec
#axiomsClean RealExecution.firstOutputState_readsUnchanged
#axiomsClean RealExecution.firstOutputState_mem
#axiomsClean RealExecution.state_mem
#axiomsClean RealExecution.source_post
#axiomsClean RealExecution.source_correct

end VeriTile.Bench.TritonBenchG.LayerNormFwd
