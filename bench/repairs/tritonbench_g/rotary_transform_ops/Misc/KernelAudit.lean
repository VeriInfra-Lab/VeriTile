import bench.repairs.tritonbench_g.rotary_transform_ops.Correct
import bench.repairs.tritonbench_g.rotary_transform_ops.Misc.TensorBridge

namespace VeriTile.Bench.Repairs.RotaryTransformOps.KernelAudit
open VeriTile.Triton

#axiomsClean KernelSource.signed_carrier
#axiomsClean TypedExecution.setup
#axiomsClean TypedExecution.indices
#axiomsClean TypedExecution.loads
#axiomsClean TypedExecution.finish
#axiomsClean TypedExecution.execution_result
#axiomsClean TypedExecution.memory_result
#axiomsClean ProgramModel.source_program
#axiomsClean sourcePrograms_order
#axiomsClean TypedGrid.complete
#axiomsClean TypedGrid.complete_order
#axiomsClean KernelTensor.lift_metadata
#axiomsClean KernelTensor.logical_read
#axiomsClean KernelIndex.data_pointer
#axiomsClean KernelIndex.coefficient_pointer
#axiomsClean KernelIndex.padded_coefficient
#axiomsClean Native.source_graph_lowering
#axiomsClean KernelEndpoints.tensor_real
#axiomsClean KernelEndpoints.tensor_native
#axiomsClean KernelEndpoints.every_token
#axiomsClean TensorBridge.rotary_transform_ops_correct
#axiomsClean rotary_transform_ops_correct_real
#axiomsClean rotary_transform_ops_kernel_fp32
#axiomsClean TensorBridge.rotary_transform_ops_fp32

#specNonCircular ProgramModel.result avoiding [FP.Structural.exec,TypedExecution.resultState]
#specNonCircular KernelTensor.bytes avoiding [FP.Structural.exec,TypedGrid.run,ProgramModel.result]
#specNonCircular realFormula avoiding [FP.Structural.exec,TypedGrid.run,ProgramModel.result,sourcePair]
#specNonCircular nativeFormula avoiding [FP.Structural.exec,TypedGrid.run,Native.graph,Native.ops]
#auditModuleAxioms
end VeriTile.Bench.Repairs.RotaryTransformOps.KernelAudit
