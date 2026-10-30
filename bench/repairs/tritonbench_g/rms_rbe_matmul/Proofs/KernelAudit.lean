import bench.repairs.tritonbench_g.rms_rbe_matmul.Correct
import bench.repairs.tritonbench_g.rms_rbe_matmul.FPEquiv
import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.KernelCounterexample

namespace VeriTile.Bench.Repairs.RmsRbeMatmul

#axiomsClean KernelCounterexample.original_ignores_rotary
#axiomsClean KernelCounterexample.missing_rotary_changes_output
#axiomsClean KernelCounterexample.original_tail_and_repaired_mask
#axiomsClean KernelSource.source_complete
#axiomsClean KernelSource.output_graph
#axiomsClean KernelGrid.offset_unique
#axiomsClean KernelGrid.covers
#axiomsClean KernelGrid.logical_complete
#axiomsClean KernelBinding.state_inputs
#axiomsClean KernelBinding.state_bytes
#axiomsClean KernelBinding.original_agreement
#axiomsClean KernelBinding.pointer_exact
#axiomsClean KernelNative.sourceDefault
#axiomsClean KernelNative.runtime_position_shape
#axiomsClean KernelNative.raw_byte_shape
#axiomsClean KernelPhysical.store_bytes
#axiomsClean KernelPhysical.original_input_frame
#axiomsClean KernelPhysical.pointer_exact
#axiomsClean KernelReal.output_real
#axiomsClean KernelRealComplete.complete
#axiomsClean KernelNativeComplete.terms_complete
#axiomsClean KernelNativeComplete.complete
#axiomsClean KernelTensorMetadata.returned
#axiomsClean rms_rbe_matmul_repaired_real_returns
#axiomsClean rms_rbe_matmul_repaired_fp_returns
#axiomsClean rms_rbe_matmul_native_addresses

#specNonCircular KernelMath.output avoiding [VeriTile.Triton.FP.Structural.exec,
  KernelSource.update,KernelSource.accumulated,KernelSource.rotated,KernelSource.scatter,
  KernelGraph.output,KernelGrid.launch]
#specNonCircular KernelGraph.output avoiding [VeriTile.Triton.FP.Structural.exec,
  KernelSource.update,KernelSource.accumulated,KernelSource.rotated,KernelSource.scatter,KernelGrid.launch]
#specNonCircular KernelBinding.originalInputs avoiding [VeriTile.Triton.FP.Structural.exec,
  KernelGraph.output,KernelSource.scatter,KernelGrid.launch]
#specNonCircular KernelRealComplete.target avoiding [VeriTile.Triton.FP.Structural.exec,
  KernelSource.inputs,KernelSource.result,KernelSource.scatter,KernelGraph.output,KernelGrid.launch]
#specNonCircular KernelNativeComplete.reference avoiding [VeriTile.Triton.FP.Structural.exec,
  KernelSource.inputs,KernelSource.update,KernelSource.accumulated,KernelSource.rotated,
  KernelSource.scatter,KernelGrid.result,KernelGrid.launch]

end VeriTile.Bench.Repairs.RmsRbeMatmul
