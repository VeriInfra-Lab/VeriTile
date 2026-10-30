import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardRun00
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardRun01
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardRun10
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardRun11
namespace VeriTile.Bench.Repairs.LayerNormOps.Forward
open VeriTile.Triton FP.Structural
noncomputable section

theorem complete_run {α : Type} [Inhabited α] (M : Algebra α)
    (p : ForwardParameters) (s : State α) :
    ∃ t, FP.Structural.exec M (forwardKernel p) s = some t ∧ t.mem = (reference M p s).mem := by
  cases hr : p.rms <;> cases ha : p.hasResidual
  · exact run00 M p s hr ha
  · exact run01 M p s hr ha
  · exact run10 M p s hr ha
  · exact run11 M p s hr ha

#axiomsClean complete_run
end
end VeriTile.Bench.Repairs.LayerNormOps.Forward
