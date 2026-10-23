import bench.repairs.tritonbench_g.rmsnorm_implementation.Proofs.RawCorrect
import bench.repairs.tritonbench_g.rmsnorm_implementation.Proofs.NativeInterface

/-! Exact retained operational statements and the separate observed native
profile. NativeInterface uses an explicit fused instruction graph; correspondence
from the split typed source to that graph remains an external compiler boundary.
No source-equivalence or Real-equality claim is inferred from this observation. -/
namespace VeriTile.Bench.Repairs.RmsnormImplementation.Auxiliary
export RawCorrect (KernelCorrect rmsnorm_kernel_correctness rmsnorm_index_commuted_correctness)
export NativeInterface (observed_kernel)
abbrev reference := NativeKernel.reference

#axiomsClean rmsnorm_kernel_correctness
#axiomsClean rmsnorm_index_commuted_correctness
#axiomsClean observed_kernel
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.RmsnormImplementation.Auxiliary
