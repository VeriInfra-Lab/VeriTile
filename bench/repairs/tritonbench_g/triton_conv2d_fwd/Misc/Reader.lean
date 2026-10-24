import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelRealResult
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelNativeResult
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelInputs
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Lowering
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Witness

namespace VeriTile.Bench.Repairs.Conv2d.Reader

#axiomsClean KernelRealResult.correct_real
#axiomsClean KernelNativeResult.fp_equivalent
#axiomsClean KernelNativeResult.fp_default
#axiomsClean KernelNativeResult.physical_execution
#axiomsClean checkCall_iff
#axiomsClean rejected_iff
#axiomsClean KernelHalf.correctness
#axiomsClean KernelDefault.correctness
#axiomsClean KernelBinding.active_iff
#axiomsClean KernelBinding.output_address
#axiomsClean KernelBinding.stores_unique
#axiomsClean KernelBinding.covers
#axiomsClean KernelGrid.launch_correct
#axiomsClean KernelOperands.input_load
#axiomsClean KernelOperands.weight_load
#axiomsClean KernelNumeric.source_reference
#axiomsClean KernelNumeric.point_reference
#axiomsClean KernelNumeric.launch_reference
#axiomsClean KernelReal.reference_real
#axiomsClean KernelNative.default_reference
#axiomsClean KernelStorage.image_initial
#axiomsClean KernelStorage.image_write
#axiomsClean KernelStorage.fold_projection
#axiomsClean KernelStores.program_bytes
#axiomsClean KernelTensor.address
#axiomsClean KernelTensor.native_read
#axiomsClean KernelEntries.real_input
#axiomsClean KernelEntries.real_weight
#axiomsClean KernelEntries.native_input
#axiomsClean KernelEntries.native_weight
#axiomsClean KernelEntries.native_initial
#axiomsClean KernelEntries.real_correct
#axiomsClean KernelEntries.native_correct
#axiomsClean Lowering.input_address
#axiomsClean Lowering.weight_address
#axiomsClean Lowering.output_address
#axiomsClean Witness.channel_tail_witness
#axiomsClean Witness.spatial_padding_witness
#specNonCircular mathematical avoiding [source,sourceValue,sourceStep,aLoad,bLoad,Host.invokeReal,KernelSource.kernel,KernelGrid.launch]
#specNonCircular KernelNumeric.reference avoiding [KernelSource.kernel,KernelGrid.value,KernelGrid.point,KernelGrid.launch,KernelStores.stores,KernelEntries.nativeHeap]
#specNonCircular KernelNative.lower32 avoiding [KernelSource.kernel,KernelGrid.value,KernelGrid.point,KernelGrid.launch]
#specNonCircular KernelNative.defaultReference avoiding [KernelSource.kernel,KernelGrid.value,KernelGrid.point,KernelGrid.launch]
#specNonCircular Domain avoiding [source,sourceValue,mathematical,ordered,Host.invoke,Host.invokeReal,KernelSource.kernel,KernelGrid.launch]
#specNonCircular KernelTensor.Binding avoiding [KernelSource.kernel,KernelGrid.launch,KernelEntries.realResult,KernelEntries.nativeResult]
#specNonCircular KernelTensor.Prepared avoiding [KernelSource.kernel,KernelGrid.launch,KernelEntries.realResult,KernelEntries.nativeResult]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.Conv2d.Reader
