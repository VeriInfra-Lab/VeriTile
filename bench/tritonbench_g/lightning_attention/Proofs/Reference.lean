import bench.tritonbench_g.lightning_attention.Reference

namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly
open VeriTile Triton FP.Structural
noncomputable section

theorem logical_derivatives (d : Host.Dimensions) : LogicalDerivatives d := by
  intro x h
  exact Target.loss_coordinate_derivatives (queryValues d x h) (keyValues d x h)
    (valueValues d x h) (incomingValues d x h)

#axiomsClean logical_derivatives
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly
