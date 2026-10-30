import bench.repairs.tritonbench_g.decay_cumsum.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.DecayCumsum.Kernels
theorem source_projects (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) :
    (source kind h p scale).toAlgorithm? = .ok (source kind h p scale).toAlgKernel := by
  cases kind <;> rfl

#axiomsClean source_projects
end VeriTile.Bench.Repairs.DecayCumsum.Kernels
