import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.layer_norm_fwd.Proofs.FirstStream
import bench.tritonbench_g.layer_norm_fwd.Proofs.Reduction

/-! Gates for completed operational phases. These gates do not claim that the
remaining source suffix, full IO contract, or all dtype profiles are complete. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd

#axiomsClean Memory.real_store
#axiomsClean Memory.bool_store
#axiomsClean Memory.scalar_store
#axiomsClean Memory.sequence_cell
#axiomsClean Memory.scatter_fold
#axiomsClean Memory.scatter_cell
#axiomsClean Memory.scatter_active
#axiomsClean Memory.scatter_frame
#axiomsClean Memory.sequence_state
#axiomsClean Execution.source_loadPrefix
#axiomsClean Execution.loadPrefix_exec
#axiomsClean Execution.source_initial_load
#axiomsClean Execution.source_scalePhase
#axiomsClean Execution.scalePhase_exec
#axiomsClean Execution.randomOp_eval
#axiomsClean Execution.keepOp_eval
#axiomsClean Execution.dropoutPhase_exec
#axiomsClean Execution.source_firstStreamPrefix
#axiomsClean Execution.source_firstStream_exec
#axiomsClean Execution.firstStreamState_x
#axiomsClean Execution.firstStreamState_mem
#axiomsClean Reduction.combined_sum
#axiomsClean Reduction.centered_square_sum
#axiomsClean Reduction.mean_op
#axiomsClean Reduction.variance_op
#axiomsClean Reduction.centered_op
#axiomsClean Reduction.rstd_op

end VeriTile.Bench.TritonBenchG.LayerNormFwd
