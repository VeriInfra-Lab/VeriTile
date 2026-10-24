import bench.repairs.tritonbench_g.sgmv_expand_slice.Correct
import bench.repairs.tritonbench_g.sgmv_expand_slice.FPEquiv

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.Reader
#axiomsClean Contract.python_checks_iff
#axiomsClean Correct.checked_real
#axiomsClean FPEquiv.checked_fp
#axiomsClean FPEquiv.checked_caller_bytes
#axiomsClean Direct.metadata_projections
#axiomsClean Direct.Bounds.formed
#axiomsClean Direct.Bounds.signed_offsets
#axiomsClean Direct.NativeCall.addresses
#axiomsClean Direct.NativeCall.metadata_spans
#axiomsClean Direct.NativeCall.output_address
#axiomsClean Direct.NativeCall.zero_batches
#axiomsClean Direct.NativeCall.caller_frame
#axiomsClean Direct.NativeCall.failure_frame
#axiomsClean Direct.NativeCall.empty
#auditModuleAxioms
end VeriTile.Bench.Repairs.SgmvExpandSlice.Reader
