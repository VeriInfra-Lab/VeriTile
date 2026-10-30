import bench.repairs.tritonbench_g.triton_attention.Whole001.Proofs.Witnesses
import bench.repairs.tritonbench_g.triton_attention.Whole001.Proofs.HostSeparate

#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.RealProgram.ds_connection
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.RealProgram.whole_memory_values
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.RealProgram.whole_memory_frame
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.RealProgram.whole_memory_inputs
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.whole_native_fp32
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Host.whole_host
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.RealProgram.dqPrefix_eq_sum
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Native.dot_coverage
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Math.tiled_sum
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Math.lowerTiledSum_eq_sum
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Math.upperTiledSum_eq_sum
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Math.whole_logical_tensors
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Math.visibleBlock_nonempty
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Math.visiblePrefix_final
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Math.sourceForwardRow_invariant
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Math.sourceForwardRow_eq_saved
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Math.sourceForward_tensors
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.offset_injective
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.backward_base_eq_forward
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.grid_coverage
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.pointer_reset
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.machine_offsets
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.preprocess_grid
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.forward_block_bounds
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.backward_block_bounds
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.backward_reset_bounds
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.row_buffer_machine_bounds
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Host.rejected
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Host.success
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Memory.writeTensorWhere_readback
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Memory.writeTensorWhere_skip
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Memory.writeTensorWhere_outside
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Math.partition_pos
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Math.probability_sum
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Math.savedMax_ge
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Math.savedMax_attained
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Math.savedNormalizer_pos
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Math.savedOutput_eq_attention
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Math.preprocessDelta_eq
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Math.sourceDS_eq_scoreGradient
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Math.scoreGradient_eq_jacobian_pullback
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Math.source_backward_eq_gradients
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Math.blockMass_rescale
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Math.blockNumerator_rescale
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Math.onlineFirst_eq_batchRow
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Math.onlineStep_eq_batchRow
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Math.onlineFold_eq_batchRow
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Math.online_forward_eq_saved
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Memory.scatter_frame
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Memory.scatter_readback
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Memory.writeTensor_readback
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Memory.writeTensor_outside
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Memory.writeRows_readback
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Memory.writeRows_outside
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Native.saved_forward
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Native.backwardInput_connection
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Native.whole_values
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Native.backwardPrefix_input
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Native.backwardPrefix_dq
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Native.backwardPrefix_dk
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Native.backwardPrefix_dv
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Native.backwardPrefix_frame
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Native.whole_frame
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Native.whole_inputs
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Native.preprocess_zeroDQ
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Native.hostOrderRun_eq_run
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Native.local_conformance_eq
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Native.whole_native
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Native.reference_witness
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Native.preprocess_coverage
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Reduction.coverage
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Reduction.sum_exact
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.nontrivial_batch_head_row
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.numerical_not_automatic
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.admitted_domain_nonempty

#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.RealProgram.whole_memory_vjp
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Math.probability_hasDerivAt
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Math.attention_hasDerivAt
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Math.loss_hasDerivAt_tangent
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.Math.attention_vjp
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.HostSeparate.primal_no_future_gradient
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.HostSeparate.forward_frame
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.HostSeparate.forward_correct
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.HostSeparate.forward_host
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.HostSeparate.preprocess_attached
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.HostSeparate.backwardSnapshot_forward
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.HostSeparate.backward_run_prefix
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.HostSeparate.backward_frame
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.HostSeparate.backward_correct
#axiomsClean VeriTile.Bench.Repairs.TritonAttention.Whole001.HostSeparate.backward_host

open Lean Elab Command in
run_cmd do
  let env ← getEnv
  let names := env.header.modules.map (fun m => Json.str m.module.toString)
  logInfo m!"TRITON_ATTENTION_READER_MODULES_JSON {(Json.arr names).compress}"
open Lean Elab Command in
run_cmd do
  let names := #[`VeriTile.Bench.Repairs.TritonAttention.Whole001.RealProgram.whole_memory_vjp,`VeriTile.Bench.Repairs.TritonAttention.Whole001.HostSeparate.forward_host,`VeriTile.Bench.Repairs.TritonAttention.Whole001.HostSeparate.backward_host,`VeriTile.Bench.Repairs.TritonAttention.Whole001.RealProgram.whole_memory_values,`VeriTile.Bench.Repairs.TritonAttention.Whole001.Native.whole_values,`VeriTile.Bench.Repairs.TritonAttention.Whole001.Host.whole_host,`VeriTile.Bench.Repairs.TritonAttention.Whole001.whole_native_fp32,`VeriTile.Bench.Repairs.TritonAttention.Whole001.Native.Instructions,`VeriTile.Bench.Repairs.TritonAttention.Whole001.Native.LocalConformance,`VeriTile.Bench.Repairs.TritonAttention.Whole001.Host.Binding,`VeriTile.Bench.Repairs.TritonAttention.Whole001.Host.InputChecks,`VeriTile.Bench.Repairs.TritonAttention.Whole001.Host.GradientChecks,`VeriTile.Bench.Repairs.TritonAttention.Whole001.Shape,`VeriTile.Bench.Repairs.TritonAttention.Whole001.Native.Owners]
  for n in names do
    let ci ← liftCoreM <| getConstInfo n
    let txt ← liftTermElabM <| Lean.Meta.ppExpr ci.type
    logInfo m!"TRITON_ATTENTION_READER_STATEMENT_JSON {(Json.mkObj [("name",Json.str n.toString),("type",Json.str txt.pretty)]).compress}"
