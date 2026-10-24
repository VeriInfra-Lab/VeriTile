import bench.repairs.tritonbench_g.rope_transform.Proofs.RawResults
import bench.repairs.tritonbench_g.rope_transform.Proofs.KernelBytes
import bench.repairs.tritonbench_g.rope_transform.Proofs.KernelAccess
import bench.repairs.tritonbench_g.rope_transform.Proofs.KernelRegressions
import bench.repairs.tritonbench_g.rope_transform.Proofs.HostCorrect
import bench.repairs.tritonbench_g.rope_transform.Proofs.HostFPEquiv

namespace VeriTile.Bench.Repairs.RopeTransform
#axiomsClean rope_forward_raw_correct
#axiomsClean rope_forward_raw_fp32
#axiomsClean KernelContract.source_binding
#axiomsClean KernelContract.sourceRun_binding
#axiomsClean KernelContract.source_success
#axiomsClean KernelContract.grid_values
#axiomsClean KernelContract.grid_frame
#axiomsClean KernelContract.input_binding
#axiomsClean KernelContract.q_address
#axiomsClean KernelContract.table_address
#axiomsClean KernelContract.access
#axiomsClean KernelContract.scalar_real
#axiomsClean KernelContract.scalar_native
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
#axiomsClean KernelContract.forward_not_backward
#axiomsClean KernelContract.fixed_coefficient_adjoint
#axiomsClean KernelContract.nonunit_not_inverse
#axiomsClean KernelContract.formed_index_counterexample
#axiomsClean Grid.different_rows
#axiomsClean rope_forward_correct
#axiomsClean rope_forward_fp32
#specNonCircular KernelContract.realValue avoiding [KernelContract.Call.source,
  VeriTile.Triton.FP.Structural.exec,KernelContract.sourceRun,Grid.run,KernelContract.physical]
#specNonCircular KernelContract.nativeValue avoiding [KernelContract.Call.source,
  VeriTile.Triton.FP.Structural.exec,KernelContract.sourceRun,Grid.run,KernelContract.byteMemory]
#specNonCircular KernelContract.Admitted avoiding [KernelContract.Call.source,
  VeriTile.Triton.FP.Structural.exec,KernelContract.sourceRun,Grid.run,
  KernelContract.realValue,KernelContract.nativeValue]
end VeriTile.Bench.Repairs.RopeTransform
