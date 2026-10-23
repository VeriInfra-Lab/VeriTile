import bench.repairs.tritonbench_g.bgmv_expand_slice.Correct
import bench.repairs.tritonbench_g.bgmv_expand_slice.FPEquiv

namespace VeriTile.Bench.Repairs.BgmvExpandSlice.Reader
#axiomsClean Contract.python_checks_iff
#axiomsClean Contract.selected_valid
#axiomsClean Correct.checked_real
#axiomsClean FPEquiv.checked_fp
#axiomsClean FPEquiv.checked_caller_bytes
#axiomsClean Direct.Bounds.active
#axiomsClean Direct.Bounds.formed
#axiomsClean Direct.Bounds.signed_offsets
#axiomsClean Direct.Native.active_addresses
#axiomsClean Direct.Native.interrupted_frame
#axiomsClean Direct.Native.failure_frame
#axiomsClean Direct.Native.metadata_span
#axiomsClean Direct.Native.empty
#specNonCircular Direct.RealExecution.formula avoiding [Direct.RealGrid.run,VeriTile.Triton.exec,Direct.Kernels.source]
#specNonCircular Direct.Native.reference avoiding [Direct.FPGrid.run,Direct.FPGrid.fold,VeriTile.Triton.FP.Structural.exec,Direct.Native.sourceStores]
#auditModuleAxioms
end VeriTile.Bench.Repairs.BgmvExpandSlice.Reader
