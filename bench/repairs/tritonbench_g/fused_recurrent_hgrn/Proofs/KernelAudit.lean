import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.KernelEndpoints
import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.KernelRealResult
import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.KernelNativeResult

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn
open VeriTile.Triton
#axiomsClean KernelRealResult.kernel_correct
#axiomsClean KernelNativeResult.kernel_fp32
#axiomsClean Execution.forward_complete
#axiomsClean Execution.backward_complete
#axiomsClean Execution.forward_program
#axiomsClean Execution.backward_program
#axiomsClean Execution.forward_grid
#axiomsClean Execution.backward_grid
#axiomsClean Checked.checked_safety
#axiomsClean Physical.key_disjoint
#axiomsClean KernelTensor.lift_read
#axiomsClean KernelTensor.coordinates
#axiomsClean KernelTensor.logical_read
#axiomsClean KernelTensor.result_eq
#axiomsClean KernelTensor.result_logical
#axiomsClean loss_derivative
#axiomsClean Kernel.real_complete
#axiomsClean Kernel.native_complete
#axiomsClean Kernel.saved_word_reload
#specNonCircular KernelTensor.Arguments avoiding [Execution.gridRun,FP.Structural.exec,state,dx,dg]
#specNonCircular Kernel.data avoiding [Execution.gridRun,FP.Structural.exec]
#specNonCircular state avoiding [Execution.gridRun,FP.Structural.exec]
#specNonCircular adjoint avoiding [Execution.gridRun,FP.Structural.exec]
#specNonCircular Native.Instructions avoiding [Execution.gridRun,FP.Structural.exec,state,dx,dg]
#specNonCircular Native.LocalConformance avoiding [Execution.gridRun,FP.Structural.exec,state,dx,dg,Native.eval]
#specNonCircular Kernel.profileSchedule avoiding [Kernel.data,Kernel.terms,Native.eval]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.FusedRecurrentHgrn
