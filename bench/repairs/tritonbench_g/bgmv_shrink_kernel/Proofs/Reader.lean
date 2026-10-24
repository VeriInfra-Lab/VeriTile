import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Correct
import bench.repairs.tritonbench_g.bgmv_shrink_kernel.FPEquiv

namespace VeriTile.Bench.Repairs.BgmvShrinkKernel.Reader
#axiomsClean Contract.python_checks_iff
#axiomsClean Correct.checked_real
#axiomsClean FPEquiv.checked_fp
#axiomsClean FPEquiv.checked_caller_bytes
#axiomsClean Direct.Workset.fiber
#axiomsClean Direct.Bounds.formed
#axiomsClean Direct.Bounds.signed_offsets
#axiomsClean Direct.NativeGraph.atomic_is_distinct
#axiomsClean Direct.Native.active_addresses
#axiomsClean Direct.Native.output_address
#axiomsClean Direct.Native.metadata_span
#axiomsClean Direct.Native.failure_frame
#axiomsClean Direct.Native.empty
#auditModuleAxioms
end VeriTile.Bench.Repairs.BgmvShrinkKernel.Reader
