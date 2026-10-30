import bench.repairs.tritonbench_g.bmm_chunk_fwd.Correct
import bench.repairs.tritonbench_g.bmm_chunk_fwd.FPEquiv
import bench.repairs.tritonbench_g.bmm_chunk_fwd.Misc.TensorBridge
import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.KernelAudit

open VeriTile.Bench.Repairs.BmmChunkFwd
#axiomsClean Correct.bmm_chunk_fwd_real
#axiomsClean FPEquiv.bmm_chunk_fwd_execution
#axiomsClean TensorBridge.bmm_chunk_fwd_tensor_real
#axiomsClean TensorBridge.bmm_chunk_fwd_native
#axiomsClean TensorBridge.bmm_chunk_fwd_kernel
#axiomsClean TensorBridge.bmm_chunk_fwd_input_tensors
#axiomsClean TensorBridge.native_launches
#guard_msgs (drop info) in
#auditModuleAxioms
