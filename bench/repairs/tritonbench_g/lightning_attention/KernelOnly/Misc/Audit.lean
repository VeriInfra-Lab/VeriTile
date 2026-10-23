import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Misc.TensorBridge
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Correct
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.FPEquiv
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Misc.RawCorrect
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Misc.RawNative

namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly
open VeriTile Triton
set_option maxHeartbeats 10000000

#axiomsClean forward_correct
#axiomsClean backward_correct
#axiomsClean forward_fp_equiv
#axiomsClean backward_fp_equiv
#axiomsClean RawCorrect.forward_correct
#axiomsClean RawCorrect.backward_correct
#axiomsClean RawNative.forward_fp_equiv
#axiomsClean RawNative.backward_fp_equiv
#axiomsClean logical_derivatives
#axiomsClean IOExecution.forward
#axiomsClean IOExecution.backward
#axiomsClean SourceSafety.invocation_safe
#axiomsClean SourceSafety.schedule_safe
#axiomsClean IOProof.forward_safety
#axiomsClean IOProof.backward_safety
#axiomsClean IOProof.forward_in_bounds
#axiomsClean IOProof.backward_in_bounds
#axiomsClean PhysicalIO.forward_application
#axiomsClean PhysicalIO.backward_application
#axiomsClean NativeMaterialization.run_typed
#axiomsClean NativeMaterialization.grid_typed
#axiomsClean NativePhysicalIO.forward_binding
#axiomsClean NativePhysicalIO.backward_binding
#axiomsClean NativePhysicalIO.forward_reference
#axiomsClean NativePhysicalIO.backward_reference
#axiomsClean NativePhysicalIO.forward_physical_binding
#axiomsClean NativePhysicalIO.backward_physical_binding
#axiomsClean NativeConnection.forward
#axiomsClean NativeConnection.backward
#axiomsClean TensorBridge.correct
#axiomsClean TensorBridge.fp_equiv
#axiomsClean forwardProgram
#axiomsClean intraProgram
#axiomsClean interProgram
#axiomsClean signed_range_invariant
#axiomsClean run_single_reverse
#axiomsClean realMasks
#axiomsClean nativeMasks
#axiomsClean stageProgram
#axiomsClean grid_execution
#axiomsClean Real.forward_correct
#axiomsClean Real.backward_correct
#axiomsClean Native.stage_execution
#axiomsClean Native.forward_native
#axiomsClean Native.backward_native
#axiomsClean Target.loss_coordinate_derivatives
#axiomsClean native_reads
#axiomsClean input_Q
#axiomsClean input_K
#axiomsClean input_V
#axiomsClean input_G
#axiomsClean decode_encode
#axiomsClean encode_decode
#axiomsClean address_unique
#axiomsClean nativeStores_word
#axiomsClean nativeStores_logical
#axiomsClean mathStores_logical
#axiomsClean writeback_readback
#axiomsClean writeback_frame
#axiomsClean writeback_inputs
#axiomsClean addresses
#axiomsClean source_rows
#axiomsClean Host.formed_intermediates
#axiomsClean Host.mask_index_machine
#axiomsClean Host.reverse_signed_row
#axiomsClean Host.reverse_pointer_step

#specNonCircular Contract avoiding [FP.Structural.exec,FP.Structural.run,TensorBridge.realExecute,TensorBridge.nativeExecute,TensorBridge.TensorPost,TensorBridge.nativeReference,mathStores,nativeStores]
#specNonCircular Preparation avoiding [FP.Structural.exec,FP.Structural.run,TensorBridge.realExecute,TensorBridge.nativeExecute,TensorBridge.TensorPost,TensorBridge.nativeReference,mathStores,nativeStores]
#specNonCircular TensorBridge.TensorPost avoiding [FP.Structural.exec,FP.Structural.run,TensorBridge.realExecute,TensorBridge.nativeExecute,mathStores,nativeStores]
#specNonCircular TensorBridge.Derivatives avoiding [FP.Structural.exec,FP.Structural.run,TensorBridge.realExecute,TensorBridge.nativeExecute,mathStores,nativeStores]
#specNonCircular Target.loss avoiding [Target.dq,Target.dk,Target.dv,FP.Structural.exec,TensorBridge.realExecute,TensorBridge.nativeExecute,mathStores,nativeStores]
#specNonCircular initialReal avoiding [TensorBridge.TensorPost,TensorBridge.Derivatives,TensorBridge.realExecute,mathStores]
#specNonCircular inputWords avoiding [TensorBridge.nativeExecute,TensorBridge.nativeReference,nativeStores]
#specNonCircular TensorBridge.nativeReference avoiding [FP.Structural.exec,FP.Structural.run,FP.Structural.step,TensorBridge.nativeExecute,Native.stageExecute,Native.gridExecute,forward,intra,inter]
#specNonCircular mathStores avoiding [TensorBridge.TensorPost,TensorBridge.Derivatives,TensorBridge.realExecute]
#specNonCircular nativeStores avoiding [TensorBridge.nativeReference,TensorBridge.nativeExecute]
#specNonCircular forwardFormula avoiding [FP.Structural.exec,FP.Structural.run,FP.Structural.runSchedule,IOExecution.forward,TensorBridge.realExecute,mathStores]
#specNonCircular backwardFormula avoiding [FP.Structural.exec,FP.Structural.run,FP.Structural.runSchedule,IOExecution.backward,TensorBridge.realExecute,mathStores]
#specNonCircular NativeReference.forward avoiding [FP.Structural.exec,FP.Structural.run,FP.Structural.runMaterializedSchedule,Native.forwardExecute,TensorBridge.nativeExecute,nativeStores]
#specNonCircular NativeReference.backward avoiding [FP.Structural.exec,FP.Structural.run,FP.Structural.runMaterializedSchedule,Native.backwardExecute,TensorBridge.nativeExecute,nativeStores]
#specNonCircular VeriTile.Bench.Repairs.LightningAttention.Native.LocalConformance avoiding [FP.Structural.exec,TensorBridge.nativeExecute,TensorBridge.nativeReference,nativeStores,Native.stageReference,VeriTile.Bench.Repairs.LightningAttention.Native.Graph.eval,VeriTile.Bench.Repairs.LightningAttention.Native.dotGraph]

/- Historical imports supply pure dimensions, target identities, and bounded
instruction definitions. These checks traverse theorem proof bodies too: the
new public results do not depend on the old normalized source or execution. -/
#specNonCircular TensorBridge.correct avoiding [VeriTile.Bench.Repairs.LightningAttention.forward,VeriTile.Bench.Repairs.LightningAttention.intra,VeriTile.Bench.Repairs.LightningAttention.inter,VeriTile.Bench.Repairs.LightningAttention.stageProgram,VeriTile.Bench.Repairs.LightningAttention.runGrid]
#specNonCircular TensorBridge.fp_equiv avoiding [VeriTile.Bench.Repairs.LightningAttention.forward,VeriTile.Bench.Repairs.LightningAttention.intra,VeriTile.Bench.Repairs.LightningAttention.inter,VeriTile.Bench.Repairs.LightningAttention.stageProgram,VeriTile.Bench.Repairs.LightningAttention.runGrid]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly
