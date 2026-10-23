import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Correct
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Misc.NativeCorrect
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Misc.TensorBridge

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast
open VeriTile.Bench.Repairs.MatmulPersistentTriton

#axiomsClean Core.prelude_invariant
#axiomsClean Core.bodyExecution
#axiomsClean Core.innerLoop
#axiomsClean Core.suffixExecution
#axiomsClean Core.outerExecution
#axiomsClean Core.programExecution
#axiomsClean Core.gridExecution
#axiomsClean Physical.materialize_frame
#axiomsClean Public.complete
#axiomsClean kernel_correct
#axiomsClean TensorBridge.correct
#axiomsClean kernel_fp_equiv
#axiomsClean TensorBridge.fp_equiv
#specNonCircular Core.executeGrid avoiding [Source.run,Source.step]
#specNonCircular Public.invoke avoiding [Source.run,Source.step]
#specNonCircular Targets.result avoiding [Core.executeGrid,Source.run,Source.step,Source.accumulator]
#specNonCircular Physical.Bindings avoiding [Targets.result,Public.invoke,Core.executeGrid]
#specNonCircular TensorBinding.Laws avoiding [Targets.result,Public.invoke,Core.executeGrid]
#specNonCircular Floating.LocalPorts avoiding [Targets.result,Public.invoke,Core.executeGrid,Source.run,Source.accumulator]
#print axioms TensorBridge.correct
#print axioms TensorBridge.fp_equiv

end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast
