import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.Misc.NativeCorrect
import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.Correct
import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.Contract
import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.Misc.TensorBridge
import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.FPEquiv
import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.NativeSource
import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.Metadata

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly
open VeriTile.Triton

#axiomsClean correct
#axiomsClean fp_equiv
#axiomsClean TensorBridge.correct
#axiomsClean TensorBridge.fp_equiv
#axiomsClean FPReference.program
#axiomsClean FPReference.whole
#axiomsClean NativeSource.source_connection
#axiomsClean Safety.projects
#axiomsClean Safety.all_formed_exact
#axiomsClean logical_input
#axiomsClean logical_output
#axiomsClean logical_metadata
#axiomsClean logical_coverage
#axiomsClean native_bytes
#axiomsClean NativeInstructions.evaluated_if_local
#axiomsClean byte_codec
#axiomsClean writeback_inputs
#axiomsClean empty_writeback

#specNonCircular Contract avoiding [FP.Structural.exec,VeriTile.Triton.exec,execute,mathematical,TensorBridge.tensorReferenceWord]
#specNonCircular mathematical avoiding [FP.Structural.exec,VeriTile.Triton.exec,execute,FPReference.output,FPReference.grid,TensorBridge.realStores,nativeStores]
#specNonCircular FPReference.output avoiding [FP.Structural.exec,FP.Structural.run,FP.Structural.step,FPReference.runGrid,kernel]
#specNonCircular TensorBridge.tensorReferenceWord avoiding [FP.Structural.exec,FP.Structural.run,FP.Structural.step,execute,nativeStores]
#specNonCircular nativeStores avoiding [TensorBridge.tensorReferenceWord,referenceWord,reference]
#specNonCircular TensorBridge.realStores avoiding [mathematical,mathematicalScore]
#specNonCircular NativeInstructions.ScalarLaws avoiding [FP.Structural.exec,execute,nativeStores,NativePhysical.evaluate,NativeInstructions.evaluate,TensorBridge.tensorReferenceWord,mathematical]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly
