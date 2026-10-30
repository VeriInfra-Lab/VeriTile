import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.Calls
import bench.repairs.tritonbench_g.chunk_retention_ops.FPEquiv
import bench.repairs.tritonbench_g.chunk_retention_ops.Correct
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.Ownership

namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
#check Source.no_cross_program_write
#axiomsClean Source.no_cross_program_write
#check Public.host_real
#check Public.host_physical_real
#check Public.mathematical_vjp
#check Public.host_native
#check Public.host_typed
#check Public.host_native_frame
#print axioms Public.host_real
#print axioms Public.host_physical_real
#print axioms Public.mathematical_vjp
#print axioms Public.host_native
#print axioms Public.host_typed
#print axioms Public.host_native_frame
#check Public.forward_call
#check Public.backward_call
#check Public.loss_gradient_polynomial
#axiomsClean Public.forward_call
#axiomsClean Public.backward_call
#axiomsClean Public.loss_gradient_polynomial
#axiomsClean MathTarget.loss_polynomial
#check Checked.chunk_retention_ops_repaired_real_returns
#check Checked.chunk_retention_ops_repaired_real_backward
#check Checked.chunk_retention_ops_repaired_fp_returns
#check Checked.chunk_retention_ops_repaired_fp_backward
#axiomsClean Checked.chunk_retention_ops_repaired_real_returns
#axiomsClean Checked.chunk_retention_ops_repaired_real_backward
#axiomsClean Checked.chunk_retention_ops_repaired_fp_returns
#axiomsClean Checked.chunk_retention_ops_repaired_fp_backward
#axiomsClean Checked.ForwardCall.final_metadata
#axiomsClean Checked.BackwardCall.initial_metadata
#axiomsClean Contract.forwardHCheck_iff
#axiomsClean Contract.forwardOCheck_iff
#axiomsClean Contract.backwardHCheck_iff
#axiomsClean Contract.backwardQKVCheck_iff
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
