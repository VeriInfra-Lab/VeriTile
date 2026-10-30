import bench.repairs.tritonbench_g.attn_fwd_triton_tile_checked.Kernels
import bench.repairs.tritonbench_g.attn_fwd_triton_tile_checked.Correct
import bench.repairs.tritonbench_g.attn_fwd_triton_tile_checked.Misc.NativeCorrect
import bench.repairs.tritonbench_g.attn_fwd_triton_tile_checked.Misc.TensorBridge

namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked
open VeriTile.Triton
#axiomsClean source_unchanged
#axiomsClean launch_iff
#axiomsClean all_tiles_fit
#axiomsClean RealBinding.agreement
#axiomsClean whole_real
#axiomsClean RawReal.whole_real
#axiomsClean RealInterface.implements
#axiomsClean Model.iteration
#axiomsClean Model.fold_exec
#axiomsClean Model.initial_exact
#axiomsClean Model.tail_read
#axiomsClean Model.program
#axiomsClean Model.reference_eq
#axiomsClean Physical.actual_input
#axiomsClean Physical.grid
#axiomsClean TensorBinding.valid_index
#axiomsClean TensorBinding.descriptor_logical
#axiomsClean Address.native_active
#axiomsClean Address.control
#axiomsClean whole_native
#axiomsClean RawNative.whole_native
#axiomsClean SourceSafety.source_safe
#axiomsClean NativeInterface.implements
#axiomsClean NativeInterface.grid_memory_eq
#axiomsClean ReadLocality.views_separated
#axiomsClean TensorBridge.whole_real
#axiomsClean TensorBridge.whole_native
#axiomsClean TensorBridge.empty_return
#specNonCircular launchChecks avoiding [whole_real,whole_native,TensorBridge.whole_real,TensorBridge.whole_native]
#specNonCircular TensorBinding.View avoiding [whole_real,whole_native,TensorBridge.whole_real,TensorBridge.whole_native,Physical.reference]
#specNonCircular Model.reference avoiding [FP.Structural.exec,FP.Structural.run,AttnFwdTriton.TypedExecution.reference]
#specNonCircular Physical.reference avoiding [FP.Structural.exec,FP.Structural.run,AttnFwdTriton.TypedExecution.reference]
#specNonCircular whole_native avoiding [AttnFwdTriton.NativeGraph.backend,AttnFwdTriton.Checked.whole_native]
#specNonCircular TensorBridge.whole_native avoiding [AttnFwdTriton.NativeGraph.backend,AttnFwdTriton.Checked.whole_native]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked
