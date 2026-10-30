import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.IOSafety
import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.IOValues

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Interface.Proof
open VeriTile Triton FP.Structural FusedRwkv6Kernel Source
noncomputable section
attribute [local instance] Classical.propDecidable
variable {α : Type} [Inhabited α]

theorem implements (M : Algebra α) (c : Shape) (hc : Sizes c) (plan : Plans c)
    (reduction : ReductionMatches M c plan) (scale : α) :
    (programIO M c scale).Implements (formula (ops M) c plan scale) := by
  intro xs _ bounds bounded s binding
  refine ⟨?_,ValueProof.values M c hc plan reduction scale xs s binding⟩
  by_cases live : Launches c
  · simpa only [programIO,if_pos live] using SafetyProof.schedule_safe M c hc scale xs bounds bounded live
      (programs c) (fun p hp => (mem_programs c p).mp hp) s
  · simp [programIO,live,ScheduleSafe]

#axiomsClean implements
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Interface.Proof
