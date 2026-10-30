import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Misc.TensorBridge
import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Correct
import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Misc.NativeCorrect

namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked
#axiomsClean Source.source_decomposition
#axiomsClean Execution.preludeExecution
#axiomsClean Execution.bodyExecution
#axiomsClean Execution.correctness
#axiomsClean Bridge.source_loads
#axiomsClean Bridge.source_value
#axiomsClean Bridge.writes_equal
#axiomsClean Bridge.programExecution
#axiomsClean Bridge.orderedExecution
#axiomsClean TensorAPI.real_target
#axiomsClean TensorAPI.native_target
#axiomsClean TensorAPI.bytes_output
#axiomsClean kernel_real
#axiomsClean TensorBridge.checked_real
#axiomsClean kernel_native
#axiomsClean TensorBridge.checked_native
#specNonCircular BmmOptimized.mathematical avoiding [Execution.exec,Bridge.grid,TensorBridge.execute]
#specNonCircular BmmOptimized.ordered avoiding [Execution.exec,Bridge.grid,TensorBridge.executeNative]
#specNonCircular TensorAPI.product avoiding [Execution.exec,Bridge.grid,TensorBridge.execute]
#specNonCircular TensorAPI.chain avoiding [Execution.exec,Bridge.grid,TensorBridge.executeNative]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked
