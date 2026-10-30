import bench.repairs.tritonbench_g.bmm_chunk_bwd.Correct
import bench.repairs.tritonbench_g.bmm_chunk_bwd.FPEquiv
import bench.repairs.tritonbench_g.bmm_chunk_bwd.Misc.TensorBridge

namespace VeriTile.Bench.Repairs.BmmChunkBwd.Review
#axiomsClean Correct.bmm_chunk_bwd_real
#axiomsClean FPEquiv.bmm_chunk_bwd_equiv
#axiomsClean PublicIO.schedule_safe
#axiomsClean PublicIO.flatten_execution
#axiomsClean KernelRealResult.bmm_chunk_bwd_real
#axiomsClean KernelInterface.versions_eq
#axiomsClean KernelInterface.native
#axiomsClean TensorBridge.bmm_chunk_bwd_real
#axiomsClean TensorBridge.bmm_chunk_bwd_equiv
#axiomsClean TensorBridge.native_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.BmmChunkBwd.Review
