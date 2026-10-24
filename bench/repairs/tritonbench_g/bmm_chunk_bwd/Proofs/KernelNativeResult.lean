import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.KernelInterface

namespace VeriTile.Bench.Repairs.BmmChunkBwd.KernelNativeResult
/- Complete raw source/candidate/native execution, output recurrence and frame.
The unchanged SM90 staging/TF32 K8 arithmetic model is NativeGraph. -/
export KernelInterface (native)
#axiomsClean native
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.BmmChunkBwd.KernelNativeResult
