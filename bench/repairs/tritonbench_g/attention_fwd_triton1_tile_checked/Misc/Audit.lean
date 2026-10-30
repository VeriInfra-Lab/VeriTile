import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Misc.NativeCorrect
import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Misc.TensorBridge
import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Proofs.Schedule

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked
open VeriTile.Triton FP.Structural

#axiomsClean Correct.kernel_real
#axiomsClean FPEquiv.kernel_native
#axiomsClean KernelIO.SourceSafety.kernel_safe
#axiomsClean KernelIO.schedule_safe
#axiomsClean KernelIO.implements
#axiomsClean PhysicalCorrect.kernel_real
#axiomsClean PhysicalFPEquiv.kernel_native
#axiomsClean NumericalCompatibility.numerical_iff
#axiomsClean Program.quoted
#axiomsClean SharedStep.body_run
#axiomsClean SharedLoop.kernel_run
#axiomsClean SharedGrid.run_success
#axiomsClean SharedGrid.output
#axiomsClean SharedGrid.history
#axiomsClean SharedGrid.frame
#axiomsClean Schedule.success
#axiomsClean Schedule.output
#axiomsClean Schedule.history
#axiomsClean Schedule.frame
#axiomsClean Projection.graph_output
#axiomsClean Projection.graph_carry
#axiomsClean Physical.success
#axiomsClean Physical.output
#axiomsClean Physical.history
#axiomsClean Physical.frame
#axiomsClean inputs_logical
#axiomsClean TensorEffect.output
#axiomsClean TensorEffect.history
#axiomsClean TensorEffect.output_identity
#axiomsClean TensorBridge.Correct.kernel_correct
#axiomsClean TensorBridge.FPEquiv.kernel_equiv
#axiomsClean TensorBridge.FPEquiv.source_fp32
#specNonCircular InputContract avoiding [SharedGrid.run,Physical.run,Native.result,TensorBridge.RealSpecification,TensorBridge.BitSpecification]
#specNonCircular TensorBinding avoiding [SharedGrid.run,Physical.run,Native.result,TensorBridge.RealSpecification,TensorBridge.BitSpecification]
#specNonCircular TensorBridge.Native.output avoiding [SharedGrid.run,Physical.run,Native.result]
#specNonCircular TensorBridge.realOutput avoiding [SharedGrid.run,Physical.run,Native.result]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked
