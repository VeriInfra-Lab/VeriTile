import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Misc.KernelResults
import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Misc.MathGrid
import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Misc.Tensor

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.TensorBridge

/-- Exact complete Tensor applications of the same `qkv_complete` source
foundation retained by KernelResults and localized by IOBytes for the public array IO. `PhysicalKernel.programTrace_complete`
connects caller memory/scalar bindings to Q/K/V states, and
`source_store_materialization` connects the resulting cells to physical stores.
The existing grid composition retains every output and exterior byte. -/
alias whole_tensor := PhysicalKernel.Ideal.whole_tensor
alias native_tensor := PhysicalKernel.native_tensor
alias coverage := PhysicalKernel.coverage
alias grid_complete := PhysicalKernel.grid_complete
alias grid_results := PhysicalKernel.grid_results
alias output_storable := Native.output_storable

#axiomsClean whole_tensor
#axiomsClean native_tensor
#axiomsClean coverage
#axiomsClean grid_complete
#axiomsClean grid_results
#axiomsClean output_storable

/-- Returned reshape storage and coordinates, together with actual Tensor pointer bounds. -/
alias result_storage := PhysicalKernel.result_storage
alias result_count := PhysicalKernel.result_count
alias head_offset := PhysicalKernel.head_offset
alias output_pointer_valid := PhysicalKernel.output_pointer_valid
alias x_load_safe := PhysicalKernel.x_load_safe
alias weight_load_safe := PhysicalKernel.weight_load_safe
alias rms_load_safe := PhysicalKernel.rms_load_safe

#axiomsClean result_storage
#axiomsClean result_count
#axiomsClean head_offset
#axiomsClean output_pointer_valid
#axiomsClean x_load_safe
#axiomsClean weight_load_safe
#axiomsClean rms_load_safe
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.TensorBridge
