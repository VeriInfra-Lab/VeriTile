import bench.repairs.tritonbench_g.parallel_attention.Proofs.HistoricalCorrect
import bench.repairs.tritonbench_g.parallel_attention.Correct
import bench.repairs.tritonbench_g.parallel_attention.FPEquiv

namespace VeriTile.Bench.Repairs.ParallelAttention
#print_spec Correct.correct_real full
#print_spec Correct.both_vjp full
#print_spec Correct.normalized_vjp full
#print_spec Correct.raw_vjp full
#print_spec FPEquiv.fp_equivalent full
#print_spec FPEquiv.returned_reference full
#print_spec FPEquiv.kernel_bytes full
#print_spec FPEquiv.caller_storage_frame full
#axiomsClean Checked.backwardCheck_iff
#axiomsClean Checked.backward_rejected_iff
#axiomsClean Checked.preparedCheck_iff
#axiomsClean Checked.incoming_lane
#axiomsClean Checked.prepared_view_address
#axiomsClean Checked.check_iff
#axiomsClean Checked.stagedPointer_actual
#axiomsClean Checked.dense_actual_strides
#axiomsClean Physical.copy_materialized
#axiomsClean Physical.observe_singleBackward
#print_spec Public.forward_correct full
#print_spec Public.both_vjp full
#print_spec Public.normalized_vjp full
#print_spec Public.raw_vjp full
#print_spec Public.fp32_forward full
#print_spec Public.fp32_returned_context full
#print_spec Public.fp32_both_context full
#print Public.query_addresses
#print Public.pointers
#print HostNative.forward_frame
#print HostNative.backward_frame
#print HostNative.single_frame
#print HostNative.forward_preserves_saved
#print HostNative.empty_forward
#print HostNative.empty_backward
#print HostNative.empty_single_backward
end VeriTile.Bench.Repairs.ParallelAttention
