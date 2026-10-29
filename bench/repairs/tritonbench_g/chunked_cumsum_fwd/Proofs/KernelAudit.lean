import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Correct
import bench.repairs.tritonbench_g.chunked_cumsum_fwd.FPEquiv

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel
#axiomsClean Correct.repaired_real
#axiomsClean FPEquiv.repaired_fp32
#axiomsClean Contract.checkInput_iff
#axiomsClean sourceRun_mem
#axiomsClean grid_correct
#axiomsClean Scalar.source_preparation
#axiomsClean Scalar.source_scan
#axiomsClean Tensor.native_input_values
#axiomsClean Tensor.accesses
#axiomsClean Tensor.byte_entry
#axiomsClean Tensor.post_store
#axiomsClean Tensor.native_complete
#axiomsClean Tensor.real_complete
#axiomsClean Numeric.clamp_special_values
#axiomsClean Numeric.arithmetic_controls

#guard_msgs (drop info) in
#specNonCircular Tensor.Inputs avoiding [source,sourceRun,launch,Tensor.NativeOutputs]
#guard_msgs (drop info) in
#specNonCircular Tensor.Buffers avoiding [source,sourceRun,launch,Tensor.NativeOutputs]
#guard_msgs (drop info) in
#specNonCircular Numeric.ScalarLaws avoiding [source,sourceRun,launch,Scalar.interpret,Tensor.NativeOutputs]
#guard_msgs (drop info) in
#specNonCircular Scalar.referenceLogicalCumulative avoiding [source,sourceRun,launch,Scalar.interpret]
#guard_msgs (drop info) in
#specNonCircular RealTarget.cumulative avoiding [source,sourceRun,launch,Scalar.interpret]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel
