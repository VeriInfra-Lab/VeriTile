import bench.repairs.tritonbench_g.sgmv_expand_slice.ByteSeparated.Correct
import bench.repairs.tritonbench_g.sgmv_expand_slice.ByteSeparated.Misc.TensorBridge

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated
open VeriTile.Triton
#axiomsClean Contract.python_checks_iff
#axiomsClean Contract.python_separate_iff
#axiomsClean Contract.disjoint_iff_no_common_byte
#axiomsClean FP32Binding.logical_read
#axiomsClean Correct.kernel_real
#axiomsClean FPEquiv.kernel_native
#axiomsClean TensorBridge.checked_real
#axiomsClean TensorBridge.checked_fp
#axiomsClean TensorBridge.checked_caller_bytes
#axiomsClean caller_frame
#axiomsClean untouched_cell_frame
#axiomsClean output_padding_frame
#axiomsClean failure_complement_frame
#axiomsClean failure_frame
#axiomsClean empty
#axiomsClean zero_batches
#axiomsClean Bounds.signed_offsets
#axiomsClean addresses
#axiomsClean output_address
#axiomsClean metadata_spans
#specNonCircular mathematicalResult avoiding [runSourceJobs,runRealJobs,FP.Structural.run]
#specNonCircular TensorBridge.nativeReference avoiding [execution,finalState,sourceStores,runSourceJobs,FP.Structural.run]
#specNonCircular TensorBridge.CompilerBinding avoiding [execution,finalState,sourceStores,runSourceJobs,FP.Structural.run]
#auditModuleAxioms
end VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated
