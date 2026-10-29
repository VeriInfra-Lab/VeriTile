import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.KernelInterface
import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.Calls

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.KernelNativeResult
open VeriTile.Triton FP.Structural
noncomputable section
open Numeric
local instance : Inhabited Word := ⟨FP.Value.ofBits .fp32 0⟩
/-- Complete forward quotation grid over raw dimensions, strides and memory, with independent recurrence outputs and full frame. -/
theorem kernel_forward_native (u : UnaryModels) (ports : ScalarPorts) (laws : ScalarLaws u ports)
    (s : Shape) (shape : s.Admitted) (p : PlanBinding s)
    (incoming finalGradient : Fin 4 → Nat) (seed final : Bool)
    (initial : State (Staged.Node Word)) :
    ∃ result,KernelInterface.run .forward (nativeArithmetic ports) s p incoming finalGradient seed final initial=some result ∧
      KernelInterface.Post .forward (arithmetic u) s p incoming finalGradient seed final initial result := by
  rw [nativeArithmetic_eq u ports laws]
  exact KernelInterface.complete .forward (arithmetic u) s shape p incoming finalGradient seed final initial

#axiomsClean kernel_forward_native

/-- Complete backward quotation grid over raw dimensions, strides and memory, with independent recurrence outputs and full frame. -/
theorem kernel_backward_native (u : UnaryModels) (ports : ScalarPorts) (laws : ScalarLaws u ports)
    (s : Shape) (shape : s.Admitted) (p : PlanBinding s)
    (incoming finalGradient : Fin 4 → Nat) (seed final : Bool)
    (initial : State (Staged.Node Word)) :
    ∃ result,KernelInterface.run .backward (nativeArithmetic ports) s p incoming finalGradient seed final initial=some result ∧
      KernelInterface.Post .backward (arithmetic u) s p incoming finalGradient seed final initial result := by
  rw [nativeArithmetic_eq u ports laws]
  exact KernelInterface.complete .backward (arithmetic u) s shape p incoming finalGradient seed final initial

#axiomsClean kernel_backward_native

end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.KernelNativeResult
