import bench.repairs.tritonbench_g.fused_rwkv6_kernel.Correct
import bench.repairs.tritonbench_g.fused_rwkv6_kernel.FPEquiv
namespace VeriTile.Bench.Repairs.FusedRwkv6Kernel.Reader
open VeriTile.Bench.Repairs.FusedRwkv6Kernel
noncomputable section
-- Importing this file checks the whole current-memory proof closure, all public
-- entries, the finite byte execution, and the concrete supported-domain witness.
#axiomsClean Correct.rwkv_real
#axiomsClean FPEquiv.rwkv_native
#axiomsClean FPEquiv.rwkv_native_if_local
#axiomsClean FPEquiv.rwkv_helper_native
#axiomsClean Checked.formed_address
#axiomsClean Checked.admit_iff
#axiomsClean Checked.geometry_iff
#axiomsClean Binding.complete
#axiomsClean correct
#axiomsClean helper_correct
#axiomsClean helper_whole_correct
#axiomsClean native_correct
#axiomsClean native_equiv_correct
#axiomsClean fpequiv
#axiomsClean unique_owner
#axiomsClean cover
#axiomsClean loop_inputs
#axiomsClean loop_carry
#axiomsClean loop_memory
#axiomsClean complete_grid
#axiomsClean returned_value
#axiomsClean terminal_value
#axiomsClean complete_frame
#axiomsClean wrapper_context
#axiomsClean helper_default
#axiomsClean reads_bounded
#axiomsClean pointer_visit
#axiomsClean byte_bound
#axiomsClean physical_inputs
#axiomsClean physical_frame
#axiomsClean byte_address_unique
#axiomsClean word_trace_mem
#axiomsClean byte_trace_realizes
#axiomsClean native_trace
#axiomsClean repaired_address
#axiomsClean original_counterexample
#axiomsClean witness_domain
#axiomsClean hoisted_bias
#axiomsClean Native.eval_agreement
#specNonCircular realState avoiding [runLoop,runProgram,runGrid,forward,physicalForward]
#specNonCircular realOutput avoiding [runLoop,runProgram,runGrid,forward,physicalForward]
#specNonCircular scalarOutput avoiding [runLoop,runProgram,runGrid,forward,physicalForward]
#print_spec correct full
#print_spec helper_whole_correct full
#print_spec native_correct full
#print_spec native_equiv_correct full
end
end VeriTile.Bench.Repairs.FusedRwkv6Kernel.Reader
