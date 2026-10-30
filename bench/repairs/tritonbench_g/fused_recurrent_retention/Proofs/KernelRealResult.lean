import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.KernelInterface

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.KernelRealResult
open VeriTile.Triton FP.Structural
noncomputable section
/-- Complete forward quotation grid over raw dimensions, strides and memory, with independent recurrence outputs and full frame. -/
theorem kernel_forward_real
    (s : Shape) (shape : s.Admitted) (p : PlanBinding s)
    (incoming finalGradient : Fin 4 → Nat) (seed final : Bool)
    (initial : State (Staged.Node ℝ)) :
    ∃ result,KernelInterface.run .forward realArithmetic s p incoming finalGradient seed final initial=some result ∧
      KernelInterface.Post .forward realArithmetic s p incoming finalGradient seed final initial result := by
  exact KernelInterface.complete .forward realArithmetic s shape p incoming finalGradient seed final initial

#axiomsClean kernel_forward_real

/-- Complete backward quotation grid over raw dimensions, strides and memory, with independent recurrence outputs and full frame. -/
theorem kernel_backward_real
    (s : Shape) (shape : s.Admitted) (p : PlanBinding s)
    (incoming finalGradient : Fin 4 → Nat) (seed final : Bool)
    (initial : State (Staged.Node ℝ)) :
    ∃ result,KernelInterface.run .backward realArithmetic s p incoming finalGradient seed final initial=some result ∧
      KernelInterface.Post .backward realArithmetic s p incoming finalGradient seed final initial result := by
  exact KernelInterface.complete .backward realArithmetic s shape p incoming finalGradient seed final initial

#axiomsClean kernel_backward_real

end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.KernelRealResult
