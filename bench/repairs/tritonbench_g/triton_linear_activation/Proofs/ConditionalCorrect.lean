import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.Execute
import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.Real
namespace VeriTile.Bench.Repairs.TritonLinearActivation
noncomputable section
/-- Whole mathematical correctness under independent fresh allocation and the
concrete admitted caller domain. No reduction/output premise is accepted. -/
specification correct (h : Allocation r caller (before : Memory ℝ)) (c : Fin 98)
    (valid : r.Valid) (currentDevice : Nat) (trials : List (Fin 98)) :
    RealPost r h (execute h realArithmetic c currentDevice trials) currentDevice := by
  rw [execute_eq h realArithmetic c currentDevice trials valid]
  exact whole_real h c valid currentDevice
#axiomsClean correct
end
end VeriTile.Bench.Repairs.TritonLinearActivation
