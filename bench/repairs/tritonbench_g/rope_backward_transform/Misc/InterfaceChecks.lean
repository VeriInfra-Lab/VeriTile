import bench.repairs.tritonbench_g.rope_backward_transform.Misc.TensorBridge
import bench.repairs.tritonbench_g.rope_backward_transform.Correct
import bench.repairs.tritonbench_g.rope_backward_transform.FPEquiv
import bench.repairs.tritonbench_g.rope_backward_transform.Proofs.KernelResults
import bench.repairs.tritonbench_g.rope_backward_transform.Proofs.KernelRegressions
import bench.repairs.tritonbench_g.rope_backward_transform.Misc.HostCorrect
import bench.repairs.tritonbench_g.rope_backward_transform.Misc.HostFPEquiv

namespace VeriTile.Bench.Repairs.RopeBackwardTransform
#axiomsClean rope_backward_kernel_correct
#axiomsClean rope_backward_kernel_fp32
#axiomsClean KernelResults.rope_backward_kernel_no_launch
#axiomsClean TensorBridge.rope_backward_kernel_correct
#axiomsClean TensorBridge.rope_backward_kernel_fp32
#axiomsClean TensorBridge.rope_backward_kernel_no_launch
#axiomsClean KernelContract.source_binding
#axiomsClean KernelContract.grid_success
#axiomsClean KernelContract.grid_values
#axiomsClean KernelContract.grid_exterior
#axiomsClean KernelContract.input_binding
#axiomsClean KernelContract.q_address
#axiomsClean KernelContract.table_address
#axiomsClean KernelContract.access
#axiomsClean KernelContract.physical_initial
#axiomsClean KernelContract.physical_store
#axiomsClean KernelContract.byte_initial
#axiomsClean KernelContract.byte_store
#axiomsClean KernelContract.source_store
#axiomsClean KernelContract.source_store_written
#axiomsClean KernelContract.native_words
#axiomsClean KernelContract.native_bytes
#axiomsClean KernelContract.bytes_frame
#axiomsClean KernelContract.table_frame
#axiomsClean KernelContract.backward_not_forward
#axiomsClean KernelContract.nonunit_not_inverse
#axiomsClean KernelContract.odd_tail
#axiomsClean KernelContract.source_index_counterexample
#axiomsClean Spec.adjoint
#axiomsClean rope_backward_correct
#axiomsClean rope_backward_fp32
#specNonCircular KernelContract.realValue avoiding [KernelContract.Call.source,
  VeriTile.Triton.FP.Structural.exec,KernelContract.Execution,grid,KernelContract.physical]
#specNonCircular KernelContract.nativeValue avoiding [KernelContract.Call.source,
  VeriTile.Triton.FP.Structural.exec,KernelContract.Execution,grid,KernelContract.byteMemory]
#specNonCircular KernelContract.Admitted avoiding [KernelContract.Call.source,
  VeriTile.Triton.FP.Structural.exec,KernelContract.Execution,grid,
  KernelContract.realValue,KernelContract.nativeValue]
end VeriTile.Bench.Repairs.RopeBackwardTransform
