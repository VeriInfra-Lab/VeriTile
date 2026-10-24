import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Misc.NativeCorrect
import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Misc.TensorBridge
import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.WordAddControls

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked
open VeriTile.Triton
open TileModel TensorModel
set_option maxHeartbeats 10000000

#axiomsClean kernel_low32
#axiomsClean native_kernel
#axiomsClean TensorBridge.kernel_tensor_low32
#axiomsClean TensorBridge.kernel_torch_low32
#axiomsClean TensorBridge.kernel_tensor_representable
#axiomsClean TensorBridge.native_tensor_lane
#axiomsClean TensorBridge.native_tensor_kernel
#axiomsClean quoted_assertion_node
#axiomsClean projection_exact
#axiomsClean projection_tail
#axiomsClean projection_length
#axiomsClean all_config_assertions
#axiomsClean true_conditional_preserved
#axiomsClean nonempty_conditional_preserved
#axiomsClean correctness
#axiomsClean tile_low32
#axiomsClean grid_low32
#axiomsClean grid_writes_unique
#axiomsClean grid_values
#axiomsClean grid_frame
#axiomsClean source_conversion
#axiomsClean zeroSeededBlock_exact
#axiomsClean modularAccumulation_dense
#axiomsClean modularAccumulation_representable
#axiomsClean nativeAccumulation_low32
#axiomsClean store_commutes
#axiomsClean state_store
#axiomsClean encode_store
#axiomsClean encode_result
#axiomsClean source_torch_input
#axiomsClean torch_source_address
#axiomsClean torch_target
#axiomsClean caller_frame
#axiomsClean admitted_indices
#axiomsClean source_a_address
#axiomsClean source_a_index_bound
#axiomsClean source_b_index_bound
#axiomsClean checkedPointers
#axiomsClean active_allocation_coverage

#specNonCircular InputContract avoiding [source,FP.Structural.exec,gridMemory,resultMemory,tensorWordTarget]
#specNonCircular FP32Binding avoiding [source,FP.Structural.exec,gridMemory,resultMemory,tensorWordTarget]
#specNonCircular tensorSignedSum avoiding [source,FP.Structural.exec,gridMemory,resultMemory,modularAccumulation]
#specNonCircular tensorWordTarget avoiding [source,FP.Structural.exec,gridMemory,resultMemory,modularAccumulation]
#specNonCircular torchSignedSum avoiding [source,FP.Structural.exec,gridMemory,resultMemory,modularAccumulation]
#specNonCircular idealRealTarget avoiding [source,FP.Structural.exec,gridMemory,resultMemory,modularAccumulation]
#specNonCircular Add32Leaf avoiding [source,FP.Structural.exec,gridMemory,resultMemory,tensorWordTarget]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked
