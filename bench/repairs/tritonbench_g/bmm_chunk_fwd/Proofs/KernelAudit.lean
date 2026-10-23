import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.RealInterface
import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.NativeInterface
import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.KernelReal

open VeriTile.Bench.Repairs.BmmChunkFwd
#axiomsClean BmmIO.real_implements
#axiomsClean BmmIO.Native.native_equivalent
#axiomsClean BmmIO.Native.native_implements
#axiomsClean KernelExecution.Raw.complete
#axiomsClean Kernels.fp32Source_eq
#axiomsClean KernelRewrite.execution
#axiomsClean KernelCandidate.native_head
#axiomsClean CleanupSource.complete
#axiomsClean CleanupReal.program
#axiomsClean CleanupReal.complete
#axiomsClean KernelReal.main_complete
#axiomsClean KernelReal.complete
#axiomsClean NativeReference.program_reference
#axiomsClean NativeReference.grid_reference
#specNonCircular Mathematical.product avoiding [Kernels.source,KernelReal.Runs,KernelReal.MainRuns,CleanupReal.schedule]
#specNonCircular NativeReference.resultGraph avoiding [Kernels.source,DirectFP.execute,DirectFP.finalState]
#guard_msgs (drop info) in
#auditModuleAxioms
