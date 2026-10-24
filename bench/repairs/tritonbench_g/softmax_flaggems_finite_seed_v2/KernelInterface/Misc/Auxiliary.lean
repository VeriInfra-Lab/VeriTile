import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.KernelInterface.Proofs.RawCorrect
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.KernelInterface.Proofs.RawFPEquiv

/-! Exact prior operational statements, retained as ordinary theorems. The
public entries use finite IO; these retain complete raw memory equality and
native observations even outside the output tensor. TensorBridge independently
retains its complete checked Tensor/bytes/backing-storage applications. -/
namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Auxiliary
namespace Correct
export RawCorrect (kernel_forward kernel_backward)
end Correct
namespace FPEquiv
export RawFPEquiv (kernel_forward kernel_backward)
end FPEquiv

#axiomsClean Correct.kernel_forward
#axiomsClean Correct.kernel_backward
#axiomsClean FPEquiv.kernel_forward
#axiomsClean FPEquiv.kernel_backward
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Auxiliary
