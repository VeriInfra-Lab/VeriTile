import bench.repairs.tritonbench_g.triton_linear_activation.IO
import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.RealInterface

namespace VeriTile.Bench.Repairs.TritonLinearActivation.LinearIO
open VeriTile.Triton FP.Structural KernelSource
noncomputable section

def logicalPreactivation (p : Parameters) (xs : LogicalInputs p)
    (i : Fin p.blockRows) (j : Fin p.blockColumns) : ℝ :=
  (if p.hasBias then xs.bias j else 0) + ∑ k : Fin p.inner,xs.left i k * xs.right j k

def logicalFormula (p : Parameters) (xs : LogicalInputs p) : LogicalOutputs p :=
  ⟨fun i j => realActivation p.activation (logicalPreactivation p xs i j),logicalPreactivation p xs⟩

theorem logical_iff (mode : Mode) (p : Parameters) (pid : Nat) :
    (logicalRealIO mode p pid).Implements (logicalFormula p) ↔
      (realIO mode p pid).Implements (realTarget mode p) := by rfl
end
end VeriTile.Bench.Repairs.TritonLinearActivation.LinearIO
