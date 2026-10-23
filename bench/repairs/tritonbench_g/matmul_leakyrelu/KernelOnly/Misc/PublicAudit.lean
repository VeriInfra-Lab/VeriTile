import bench.repairs.tritonbench_g.matmul_leakyrelu.KernelOnly.Misc.CandidateCorrect
import bench.repairs.tritonbench_g.matmul_leakyrelu.KernelOnly.Misc.TensorBridge

namespace VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly
open VeriTile.Triton

#axiomsClean request_eq
#axiomsClean input_address
#axiomsClean logical_fp32
#axiomsClean input_view_in_backing
#axiomsClean output_pointer
#axiomsClean admitted
#axiomsClean tensorContract
#axiomsClean scheduled_grid
#axiomsClean empty_grid
#axiomsClean execute
#axiomsClean source_store_address
#axiomsClean tensor_result
#axiomsClean backing_frame
#axiomsClean mathematical_kernel_raw
#axiomsClean floating_kernel_raw
#axiomsClean TensorBridge.kernel_request
#axiomsClean TensorBridge.half_reconstruct
#axiomsClean TensorBridge.actual_half_result
#axiomsClean TensorBridge.mathematical_kernel
#axiomsClean TensorBridge.floating_kernel
#specNonCircular TensorBridge.mathematical avoiding [run, dsl, sourceDsl, initial, writeback]
#specNonCircular TensorBridge.floating avoiding [run, dsl, sourceDsl, initial, writeback]
#specNonCircular writeback avoiding [TensorBridge.mathematical, TensorBridge.floating]

end VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly
