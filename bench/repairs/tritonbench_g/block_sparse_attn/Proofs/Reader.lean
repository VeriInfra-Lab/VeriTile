import bench.repairs.tritonbench_g.block_sparse_attn.Proofs.Witnesses
import bench.repairs.tritonbench_g.block_sparse_attn.Correct

#print VeriTile.Bench.Repairs.BlockSparseAttn.whole_real
#print VeriTile.Bench.Repairs.BlockSparseAttn.whole_native
#print VeriTile.Bench.Repairs.BlockSparseAttn.whole_fp
#print VeriTile.Bench.Repairs.BlockSparseAttn.whole_sm90_b16
#print VeriTile.Bench.Repairs.BlockSparseAttn.WholeReal
#print VeriTile.Bench.Repairs.BlockSparseAttn.WholeNative
#print VeriTile.Bench.Repairs.BlockSparseAttn.Layout
#print VeriTile.Bench.Repairs.BlockSparseAttn.Reduction
#print VeriTile.Bench.Repairs.BlockSparseAttn.Ops
#print VeriTile.Bench.Repairs.BlockSparseAttn.Native.Instructions
#print VeriTile.Bench.Repairs.BlockSparseAttn.Native.Correspondence
#print VeriTile.Bench.Repairs.BlockSparseAttn.Host.TensorGuard
#print VeriTile.Bench.Repairs.BlockSparseAttn.Host.Contract
#print VeriTile.Bench.Repairs.BlockSparseAttn.Host.Allocation
#print VeriTile.Bench.Repairs.BlockSparseAttn.Host.Arguments
#print VeriTile.Bench.Repairs.BlockSparseAttn.Host.MetadataBound
#print VeriTile.Bench.Repairs.BlockSparseAttn.Safety.DataContract
#print VeriTile.Bench.Repairs.BlockSparseAttn.Safety.dataNodes
#print VeriTile.Bench.Repairs.BlockSparseAttn.SM90B16.Binding
#print VeriTile.Bench.Repairs.BlockSparseAttn.SM90B16.plan
#print VeriTile.Bench.Repairs.BlockSparseAttn.normalizedStep
#print VeriTile.Bench.Repairs.BlockSparseAttn.blockStep
#print VeriTile.Bench.Repairs.BlockSparseAttn.Spec.output

#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.channel_bound
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.csr_first_pointer
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.csr_last_pointer
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.query_grid_i32
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.output_grid_i32
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.whole_real
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.whole_fp
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.whole_native
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.whole_sm90_b16
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.channel_coverage
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.lane_coverage
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.lane_injective
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.target_injective
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.target_bound
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.csr_visit
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.csr_stop
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Host.tensor_bound
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Host.tensor_access
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Host.tensor_formed
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Host.row_envelope
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Host.query_access
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Host.output_address
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Host.output_access
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Host.complete
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Host.identity
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Host.output
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Host.frame
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.row_readonly
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.execute_readonly
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.written_value
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.schedule_mem
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.schedule_nodup
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.call_value
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.call_frame
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Native.correspondence_eq
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Native.reflexive
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Native.execute_load
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Native.output_bytes
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Native.byte_frame
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Real.dot_some
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Real.reduce_sum
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Real.reduce_max
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Real.scoreDot_real
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Real.score_real
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Real.v_real
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Real.fold_real
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Real.weighted_raw
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Real.row_real
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Real.whole_real
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Real.sumReal_fin
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Real.dot_fin
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Real.normalized_real
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.SM90B16.coverage
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Safety.signed32_exact
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Safety.byte_allocated
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Safety.byte_signed64
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Safety.all_data_nodes_exact
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Safety.csr_readonly
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Safety.grouped_head_bound
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Source.lanes_mem
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Source.lanes_nodup
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Source.store_mask
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Source.work_complete
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Source.query_pointer
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Source.key_pointer
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Source.value_pointer
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Source.output_pointer
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Spec.scoreLoop_full
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Spec.recurrence_output
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Spec.empty_output
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Witnesses.numeric_not_automatic
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Witnesses.shape_valid
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Witnesses.last_channel_covered
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Witnesses.last_channel_active
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Witnesses.final_output_address
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Witnesses.stride_two_channel31
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Witnesses.fourth_block_value
#axiomsClean VeriTile.Bench.Repairs.BlockSparseAttn.Witnesses.codec_nan_payload

open Lean Elab Command in
run_cmd do
  let env ← getEnv
  let names := env.header.modules.map (fun m => Json.str m.module.toString)
  logInfo m!"BLOCK_SPARSE_READER_MODULES_JSON {(Json.arr names).compress}"
open Lean Elab Command in
run_cmd do
  let names := #[`whole_real,`whole_native,`whole_fp,`whole_sm90_b16]
  for short in names do
    let n := `VeriTile.Bench.Repairs.BlockSparseAttn ++ short
    let ci ← liftCoreM <| getConstInfo n
    let txt ← liftTermElabM <| Lean.Meta.ppExpr ci.type
    logInfo m!"BLOCK_SPARSE_READER_STATEMENT_JSON {(Json.mkObj [("name",Json.str n.toString),("type",Json.str txt.pretty)]).compress}"
