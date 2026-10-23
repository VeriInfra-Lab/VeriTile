import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Contract
import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Correct
import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Misc.NativeCorrect
import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Misc.TensorBridge
import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Misc.NativeCorrespondence
import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Misc.Metadata

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly
open VeriTile.Triton
open TensorBridge

#axiomsClean kernel_real
#axiomsClean kernel_native

#axiomsClean correct
#axiomsClean fp_equiv
#axiomsClean FPReference.program
#axiomsClean FPReference.whole
#axiomsClean NativeSource.source_connection
#axiomsClean NativeSource.native_source
#axiomsClean NativeSource.direct_source_control
#axiomsClean NativeSigned.ending_exact
#axiomsClean NativeSigned.key_mask
#axiomsClean NativeSigned.causal_mask
#axiomsClean NativeSigned.payload_offset
#axiomsClean Safety.machine
#axiomsClean Safety.query_formed
#axiomsClean Safety.all_formed
#axiomsClean Safety.page_formed
#axiomsClean logical_input
#axiomsClean logical_output
#axiomsClean logical_metadata
#axiomsClean source_metadata
#axiomsClean index_native_pointer
#axiomsClean metadata_byte_offset
#axiomsClean logical_coverage
#axiomsClean native_bytes
#axiomsClean NativeInstructions.evaluated_if_local
#axiomsClean byte_codec
#axiomsClean writeback_inputs
#axiomsClean empty_writeback

#specNonCircular Contract avoiding [FP.Structural.exec,VeriTile.Triton.exec,execute,mathematical,tensorReferenceWord]
#specNonCircular mathematical avoiding [FP.Structural.exec,VeriTile.Triton.exec,execute,FPReference.output,FPReference.grid,realStores,nativeStores]
#specNonCircular realInitial avoiding [mathematical,mathematicalInputs]
#specNonCircular FPReference.output avoiding [FP.Structural.exec,FP.Structural.run,FP.Structural.step,FPSuccess.runGrid,kernel]
#specNonCircular tensorReferenceWord avoiding [FP.Structural.exec,FP.Structural.run,FP.Structural.step,execute,nativeStores]
#specNonCircular nativeStores avoiding [tensorReferenceWord,referenceWord,reference]
#specNonCircular realStores avoiding [mathematical,mathematicalInputs]
#specNonCircular NativeInstructions.ScalarLaws avoiding [FP.Structural.exec,execute,nativeStores,NativeGraph.evaluate,NativeInstructions.evaluate,tensorReferenceWord,mathematical]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly
